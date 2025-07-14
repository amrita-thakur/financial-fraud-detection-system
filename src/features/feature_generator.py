"""
Feature Generator for Financial Fraud Detection
Reads transaction data and generates ML-ready features.
"""

import os
import argparse
from logging.handlers import TimedRotatingFileHandler

import pandas as pd
import numpy as np
import logging
from typing import Optional


def generate_features(transactions_df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Generate additional features for ML model training."""
    df = transactions_df.copy()
    # Parse timestamp if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    # Amount-based features
    df['amount_log'] = np.log(df['amount'] + 1)
    if 'user_avg_amount' in df.columns:
        df['amount_relative_to_avg'] = df['amount'] / df['user_avg_amount']
        df['is_high_amount'] = (df['amount'] > df['user_avg_amount'] * 3).astype(int)
    # Categorical encoding
    categorical_cols = ['merchant_category', 'device_type', 'user_pattern', 'merchant_location']
    for col in categorical_cols:
        if col in df.columns:
            # Handle any problematic values in categorical columns
            df[col] = df[col].astype(str).fillna('unknown')
            # Use get_dummies to create one-hot encoded columns
            dummies = pd.get_dummies(df[col], prefix=col[:3])
            # Drop the original column and add the dummy columns
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            if logger:
                logger.info(f"Encoded {col} into {len(dummies.columns)} dummy columns")
    
    # Remove any remaining object columns that might cause issues
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(object_cols) > 0:
        if logger:
            logger.info(f"Removing remaining object columns: {object_cols}")
            logger.warning(f"These columns will be dropped: {object_cols}")
        df = df.drop(columns=object_cols)
    
    # Final check - ensure no object columns remain
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(remaining_object_cols) > 0:
        if logger:
            logger.error(f"Still have object columns after processing: {remaining_object_cols}")
        # Force drop any remaining object columns
        df = df.drop(columns=remaining_object_cols)
    if logger:
        logger.info(f"Generated features for {len(df)} transactions.")
    return df


if __name__ == "__main__":
    

    # Logger setup (only for CLI usage)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'feature_generator.log')
    logger = logging.getLogger("feature_generator")
    logger.setLevel(logging.INFO)
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    file_handler.suffix = "%Y-%m-%d"
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description="Generate features from transaction data.")
    parser.add_argument('--input', type=str, required=True, help='Path to input transactions CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output features CSV')
    args = parser.parse_args()

    logger.info(f"Reading transactions from {args.input}")
    transactions_df = pd.read_csv(args.input)
    logger.info(f"Generating features for {len(transactions_df)} transactions...")
    features_df = generate_features(transactions_df, logger=logger)
    features_df.to_csv(args.output, index=False)
    logger.info(f"Features saved to {args.output}")
    print(f"Features generated and saved to {args.output}") 
