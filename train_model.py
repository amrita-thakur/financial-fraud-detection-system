#!/usr/bin/env python3
"""
Standalone script to train fraud detection models.
Run this script to train all models with hyperparameter tuning.
"""

import sys
import os
import logging

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model_trainer import FraudDetectionModelTrainer

def main():
    """Main function to train models."""
    # Create logs directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging with both file and console handlers
    from datetime import datetime
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
    
    try:
        # Initialize trainer with fast models only (skip slow SVM and Neural Network)
        logger.info("Initializing trainer with fast models only (skipping SVM and Neural Network)...")
        trainer = FraudDetectionModelTrainer(skip_svm=True, skip_neural_network=True)
        
        # Verify which models will be trained
        models_to_train = list(trainer.model_configs.keys())
        logger.info(f"Models to train: {models_to_train}")
        logger.info(f"Expected training time: 3-8 minutes for {len(models_to_train)} models")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        X, y = trainer.load_and_prepare_data()
        
        # Train models
        logger.info("Training models with hyperparameter tuning...")
        results = trainer.train_models(X, y)
        
        # Save models
        logger.info("Saving trained models...")
        trainer.save_models()
        
        # Generate plots and report
        logger.info("Generating evaluation plots and report...")
        trainer.plot_results(results)
        report_path = trainer.generate_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show model rankings
        print("\nModel Rankings (by Test AUC):")
        print("-" * 40)
        
        sorted_models = sorted(results.items(), key=lambda x: x[1].get('test_auc', 0), reverse=True)
        
        for i, (model_name, result) in enumerate(sorted_models, 1):
            if 'error' not in result:
                print(f"{i}. {model_name.upper()}")
                print(f"   Test AUC: {result['test_auc']:.4f}")
                print(f"   Test AP: {result['test_ap']:.4f}")
                print(f"   Best CV Score: {result['best_score']:.4f}")
            else:
                print(f"{i}. {model_name.upper()} - ERROR: {result['error']}")
        
        best_model = sorted_models[0]
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   AUC: {best_model[1]['test_auc']:.4f}")
        print(f"   AP: {best_model[1]['test_ap']:.4f}")
        
        print(f"\n📊 Training report saved to: {report_path}")
        print(f"📁 Models saved to: models/ directory")
        print(f"📝 Log file: {log_file}")
        
        print("\nNext steps:")
        print("1. Evaluate models: python src/models/model_evaluator.py")
        print("2. Test predictions: python src/models/model_predictor.py")
        print("3. Start API server: python src/api/main.py")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        print(f"\n❌ Model training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
