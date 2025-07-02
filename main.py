"""
Financial Fraud Detection System - Main Entry Point
Hybrid approach combining traditional ML models with LLM/Agent-based AI
"""

import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_generator import FraudDataGenerator, logger
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the fraud detection system."""
    logger.info("Starting Financial Fraud Detection System...")
    
    # Initialize data generator
    generator = FraudDataGenerator(seed=42)
    
    # Generate dataset
    logger.info("Generating synthetic fraud detection dataset...")
    users_df, transactions_df, features_df = generator.generate_dataset(
        num_users=10000,
        num_transactions=100000
    )
    
    # Save dataset (use the original transactions DataFrame, not features)
    generator.save_dataset(users_df, transactions_df, "data")
    
    logger.info("Dataset generation completed!")
    logger.info(f"Generated {len(users_df)} users and {len(transactions_df)} transactions")
    
    # Print sample statistics
    fraud_rate = transactions_df['is_fraud'].mean()
    logger.info(f"Overall fraud rate: {fraud_rate:.2%}")
    
    print("\n" + "="*50)
    print("FRAUD DETECTION SYSTEM - DATASET GENERATED")
    print("="*50)
    print(f"Users: {len(users_df):,}")
    print(f"Transactions: {len(transactions_df):,}")
    print(f"Fraud Rate: {fraud_rate:.2%}")
    print("\nFraud Reasons Distribution:")
    print(transactions_df['fraud_reason'].value_counts())
    print("\nNext steps:")
    print("1. Run feature engineering: python src/features/feature_engineering.py")
    print("2. Train ML model: python src/models/train_model.py")
    print("3. Start API server: python src/api/main.py")


if __name__ == "__main__":
    main() 