"""
Model Trainer for Financial Fraud Detection
Trains multiple ML models with hyperparameter tuning and evaluation.
"""

import os
import joblib
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    GridSearchCV,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class FraudDetectionModelTrainer:
    """Trainer for fraud detection models with comprehensive evaluation."""
    
    def __init__(self, data_path: str = "data/features.csv", models_dir: str = "models", skip_svm: bool = False, skip_neural_network: bool = False):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the features CSV file
            models_dir: Directory to save trained models
            skip_svm: If True, skip SVM training (slowest model)
            skip_neural_network: If True, skip Neural Network training (slow model)
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.skip_svm = skip_svm
        self.skip_neural_network = skip_neural_network
        self.logger = logging.getLogger(__name__)
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.best_params = {}
        self.cv_scores = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear'],
                    'class_weight': ['balanced']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True, cache_size=1000),
                'params': {
                    'C': [1],  # Reduced from [1, 10] to just [1]
                    'kernel': ['rbf'],
                    'gamma': ['scale'],
                    'class_weight': ['balanced']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=200),  # Reduced from 500 to 200
                'params': {
                    'hidden_layer_sizes': [(50,)],  # Reduced from [(50,), (100,)] to just [(50,)]
                    'alpha': [0.01],  # Reduced from [0.001, 0.01] to just [0.01]
                    'learning_rate': ['adaptive'],
                    'early_stopping': [True]
                }
            }
        }
        
        # Remove SVM if skip_svm is True
        if self.skip_svm:
            self.logger.info("Skipping SVM training (slowest model)")
            del self.model_configs['svm']
        
        # Remove Neural Network if skip_neural_network is True
        if self.skip_neural_network:
            self.logger.info("Skipping Neural Network training (slow model)")
            del self.model_configs['neural_network']
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for training."""
        self.logger.info(f"Loading data from {self.data_path}")
        
        # Load features
        df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Separate features and target
        target_col = 'is_fraud'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Remove non-feature columns
        exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud', 'fraud_reason']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X: pd.DataFrame = df[feature_cols].copy()
        y: pd.Series = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Identify and handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        self.logger.info(f"Found categorical columns: {list(categorical_cols)}")
        
        # Encode categorical variables
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                self.logger.info(f"Encoded categorical column: {col}")
            except Exception as e:
                self.logger.warning(f"Error encoding {col}: {str(e)}")
                # Drop problematic categorical columns
                X = X.drop(columns=[col])
                self.logger.info(f"Dropped problematic column: {col}")
        
        # Scale only numerical features (excluding encoded categoricals)
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        self.logger.info(f"Scaling numerical columns: {list(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            self.scalers['numerical'] = scaler
        
        self.logger.info(f"Prepared {len(X.columns)} features for training")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models with hyperparameter tuning."""
        self.logger.info("Starting model training with hyperparameter tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Special handling for SVM - use smaller dataset
                if model_name == 'svm':
                    self.logger.info("SVM detected - using smaller dataset for faster training...")
                    # Use only 20% of training data for SVM to speed up training
                    X_train_svm, _, y_train_svm, _ = train_test_split(
                        X_train, y_train, train_size=0.2, random_state=42, stratify=y_train
                    )
                    self.logger.info(f"Using {len(X_train_svm)} samples for SVM training (instead of {len(X_train)})")
                    X_train_current = X_train_svm
                    y_train_current = y_train_svm
                elif model_name == 'neural_network':
                    self.logger.info("Neural Network detected - using smaller dataset for faster training...")
                    # Use only 30% of training data for Neural Network to speed up training
                    X_train_nn, _, y_train_nn, _ = train_test_split(
                        X_train, y_train, train_size=0.3, random_state=42, stratify=y_train
                    )
                    self.logger.info(f"Using {len(X_train_nn)} samples for Neural Network training (instead of {len(X_train)})")
                    X_train_current = X_train_nn
                    y_train_current = y_train_nn
                else:
                    X_train_current = X_train
                    y_train_current = y_train
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['params'],
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Reduced from 5 to 3
                    scoring='roc_auc',
                    n_jobs=1,  # Use single core to prevent hanging
                    verbose=1  # Add progress output
                )
                
                grid_search.fit(X_train_current, y_train_current)
                
                # Store best model and parameters
                self.models[model_name] = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
                
                # Evaluate on test set
                y_pred = grid_search.predict(X_test)
                y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_auc': roc_auc_score(y_test, y_pred_proba),
                    'test_ap': average_precision_score(y_test, y_pred_proba),
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                self.logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
                self.logger.info(f"{model_name} - Test AUC: {results[model_name]['test_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def save_models(self):
        """Save trained models and preprocessing objects."""
        self.logger.info("Saving trained models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.joblib")
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers and encoders
        preprocessing_path = os.path.join(self.models_dir, f"preprocessing_{timestamp}.joblib")
        preprocessing_data = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders
        }
        joblib.dump(preprocessing_data, preprocessing_path)
        self.logger.info(f"Saved preprocessing objects to {preprocessing_path}")
        
        # Save best parameters
        params_path = os.path.join(self.models_dir, f"best_params_{timestamp}.json")
        import json
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        self.logger.info(f"Saved best parameters to {params_path}")
    
    def plot_results(self, results: Dict[str, Any], save_path: str = "models"):
        """Plot and save model comparison results."""
        os.makedirs(save_path, exist_ok=True)
        
        # Model comparison plot
        model_names = list(results.keys())
        auc_scores = [results[name].get('test_auc', 0) for name in model_names]
        
        plt.figure(figsize=(12, 8))
        
        # AUC comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(model_names, auc_scores, color='skyblue')
        plt.title('Model AUC Comparison')
        plt.ylabel('AUC Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Precision-Recall curves
        plt.subplot(2, 2, 2)
        for model_name in model_names:
            if 'error' not in results[model_name]:
                # Load test data for plotting
                X, y = self.load_and_prepare_data()
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                y_pred_proba = self.models[model_name].predict_proba(X_test)[:, 1]
                
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.plot(recall, precision, label=model_name, linewidth=2)
        
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ROC curves
        plt.subplot(2, 2, 3)
        for model_name in model_names:
            if 'error' not in results[model_name]:
                X, y = self.load_and_prepare_data()
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                y_pred_proba = self.models[model_name].predict_proba(X_test)[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={results[model_name]['test_auc']:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title('ROC Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrices
        plt.subplot(2, 2, 4)
        best_model = max(results.items(), key=lambda x: x[1].get('test_auc', 0))[0]
        if 'error' not in results[best_model]:
            cm = results[best_model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved model comparison plot to {plot_path}")
    
    def generate_report(self, results: Dict[str, Any], save_path: str = "models"):
        """Generate a comprehensive training report."""
        os.makedirs(save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_path, f"training_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FINANCIAL FRAUD DETECTION - MODEL TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n\n")
            
            # Model rankings
            f.write("MODEL RANKINGS (by Test AUC):\n")
            f.write("-" * 40 + "\n")
            
            sorted_models = sorted(results.items(), key=lambda x: x[1].get('test_auc', 0), reverse=True)
            
            for i, (model_name, result) in enumerate(sorted_models, 1):
                if 'error' not in result:
                    f.write(f"{i}. {model_name.upper()}\n")
                    f.write(f"   Test AUC: {result['test_auc']:.4f}\n")
                    f.write(f"   Test AP: {result['test_ap']:.4f}\n")
                    f.write(f"   Best CV Score: {result['best_score']:.4f}\n")
                    f.write(f"   Best Parameters: {result['best_params']}\n\n")
                else:
                    f.write(f"{i}. {model_name.upper()} - ERROR: {result['error']}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 40 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name.upper()}:\n")
                f.write("-" * 20 + "\n")
                
                if 'error' not in result:
                    f.write(f"Best Parameters: {result['best_params']}\n")
                    f.write(f"Cross-Validation AUC: {result['best_score']:.4f}\n")
                    f.write(f"Test AUC: {result['test_auc']:.4f}\n")
                    f.write(f"Test Average Precision: {result['test_ap']:.4f}\n\n")
                    
                    f.write("Classification Report:\n")
                    f.write(result['classification_report'])
                    f.write("\n")
                    
                    f.write("Confusion Matrix:\n")
                    f.write(str(result['confusion_matrix']))
                    f.write("\n\n")
                else:
                    f.write(f"ERROR: {result['error']}\n\n")
        
        self.logger.info(f"Generated training report: {report_path}")
        return report_path


def main():
    """Main function to run model training."""
    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging with both file and console handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"model_training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Console output
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Financial Fraud Detection Model Training...")
    logger.info(f"Log file: {log_file}")
    
    # SVM optimization note
    logger.info("SVM Optimization: Using 20% of training data and reduced hyperparameters for faster training")
    logger.info("Neural Network Optimization: Using 30% of training data and reduced hyperparameters for faster training")
    logger.info("To skip slow models:")
    logger.info("  - Skip SVM: FraudDetectionModelTrainer(skip_svm=True)")
    logger.info("  - Skip Neural Network: FraudDetectionModelTrainer(skip_neural_network=True)")
    logger.info("  - Skip both: FraudDetectionModelTrainer(skip_svm=True, skip_neural_network=True)")
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer()  # Set skip_svm=True, skip_neural_network=True to skip slow models
    
    try:
        # Load and prepare data
        X, y = trainer.load_and_prepare_data()
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Save models
        trainer.save_models()
        
        # Generate plots and report
        trainer.plot_results(results)
        report_path = trainer.generate_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED")
        print("="*60)
        
        best_model = max(results.items(), key=lambda x: x[1].get('test_auc', 0))
        print(f"Best Model: {best_model[0]}")
        print(f"Best AUC: {best_model[1]['test_auc']:.4f}")
        print(f"Best AP: {best_model[1]['test_ap']:.4f}")
        
        print(f"\nModels saved to: {trainer.models_dir}")
        print(f"Training report: {report_path}")
        print(f"Log file: {log_file}")
        
        print("\nNext steps:")
        print("1. Evaluate model performance: python src/models/evaluate_model.py")
        print("2. Start API server: python src/api/main.py")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 
