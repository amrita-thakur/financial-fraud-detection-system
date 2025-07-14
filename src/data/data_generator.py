"""
Synthetic Financial Transaction Data Generator for Fraud Detection
Generates realistic transaction data with various fraud patterns for ML model training and inference.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from typing import Tuple
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os

# Custom logger setup
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'fraud_detection.log')

logger = logging.getLogger("fraud_detection")
logger.setLevel(logging.INFO)

# File handler: rotates daily, keeps 7 days of logs
file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
file_handler.suffix = "%Y-%m-%d"

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))

# Avoid duplicate handlers 
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Remove the old root logger config
# logging.basicConfig(level=logging.INFO)


class FraudDataGenerator:
    """
    Generates synthetic financial transaction data with realistic fraud patterns.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Merchant categories and their typical transaction amounts
        self.merchant_categories = {
            'electronics': {'min_amount': 50, 'max_amount': 2000, 'fraud_rate': 0.08},
            'clothing': {'min_amount': 20, 'max_amount': 500, 'fraud_rate': 0.05},
            'food_dining': {'min_amount': 10, 'max_amount': 200, 'fraud_rate': 0.03},
            'gas_station': {'min_amount': 20, 'max_amount': 100, 'fraud_rate': 0.06},
            'grocery': {'min_amount': 15, 'max_amount': 300, 'fraud_rate': 0.02},
            'travel': {'min_amount': 100, 'max_amount': 5000, 'fraud_rate': 0.12},
            'gaming': {'min_amount': 5, 'max_amount': 200, 'fraud_rate': 0.15},
            'healthcare': {'min_amount': 25, 'max_amount': 1000, 'fraud_rate': 0.04},
            'entertainment': {'min_amount': 15, 'max_amount': 150, 'fraud_rate': 0.07},
            'utilities': {'min_amount': 50, 'max_amount': 500, 'fraud_rate': 0.01}
        }
        
        # Geographic locations with fraud risk levels
        self.locations = {
            'New York, NY': {'fraud_rate': 0.08, 'timezone': 'EST'},
            'Los Angeles, CA': {'fraud_rate': 0.07, 'timezone': 'PST'},
            'Chicago, IL': {'fraud_rate': 0.06, 'timezone': 'CST'},
            'Houston, TX': {'fraud_rate': 0.05, 'timezone': 'CST'},
            'Phoenix, AZ': {'fraud_rate': 0.06, 'timezone': 'MST'},
            'Philadelphia, PA': {'fraud_rate': 0.07, 'timezone': 'EST'},
            'San Antonio, TX': {'fraud_rate': 0.04, 'timezone': 'CST'},
            'San Diego, CA': {'fraud_rate': 0.05, 'timezone': 'PST'},
            'Dallas, TX': {'fraud_rate': 0.06, 'timezone': 'CST'},
            'San Jose, CA': {'fraud_rate': 0.05, 'timezone': 'PST'},
            'Austin, TX': {'fraud_rate': 0.04, 'timezone': 'CST'},
            'Jacksonville, FL': {'fraud_rate': 0.06, 'timezone': 'EST'},
            'Fort Worth, TX': {'fraud_rate': 0.05, 'timezone': 'CST'},
            'Columbus, OH': {'fraud_rate': 0.05, 'timezone': 'EST'},
            'Charlotte, NC': {'fraud_rate': 0.05, 'timezone': 'EST'},
            'San Francisco, CA': {'fraud_rate': 0.07, 'timezone': 'PST'},
            'Indianapolis, IN': {'fraud_rate': 0.04, 'timezone': 'EST'},
            'Seattle, WA': {'fraud_rate': 0.06, 'timezone': 'PST'},
            'Denver, CO': {'fraud_rate': 0.05, 'timezone': 'MST'},
            'Washington, DC': {'fraud_rate': 0.08, 'timezone': 'EST'}
        }
        
        # Device types and their risk levels
        self.device_types = {
            'mobile_web': {'fraud_rate': 0.06},
            'mobile_app': {'fraud_rate': 0.04},
            'desktop_web': {'fraud_rate': 0.05},
            'desktop_app': {'fraud_rate': 0.03},
            'tablet_web': {'fraud_rate': 0.05},
            'tablet_app': {'fraud_rate': 0.04}
        }
        
        # User behavior patterns
        self.user_patterns = {
            'normal': {'avg_daily_transactions': 2.5, 'avg_amount': 85, 'fraud_rate': 0.02},
            'high_value': {'avg_daily_transactions': 1.2, 'avg_amount': 450, 'fraud_rate': 0.04},
            'frequent': {'avg_daily_transactions': 8.0, 'avg_amount': 35, 'fraud_rate': 0.03},
            'new_user': {'avg_daily_transactions': 1.8, 'avg_amount': 120, 'fraud_rate': 0.08}
        }

    def generate_user_profiles(self, num_users: int) -> pd.DataFrame:
        """Generate user profiles with different behavior patterns."""
        users = []
        
        for i in range(num_users):
            user_id = f"USER_{str(uuid.uuid4())[:8].upper()}"
            pattern = random.choices(
                list(self.user_patterns.keys()),
                weights=[0.6, 0.2, 0.15, 0.05]  # More normal users, fewer new users
            )[0]
            
            user = {
                'user_id': user_id,
                'pattern': pattern,
                'avg_daily_transactions': self.user_patterns[pattern]['avg_daily_transactions'],
                'avg_amount': self.user_patterns[pattern]['avg_amount'],
                'fraud_rate': self.user_patterns[pattern]['fraud_rate'],
                'created_date': datetime.now() - timedelta(days=random.randint(30, 365)),
                'location': random.choice(list(self.locations.keys())),
                'device_preference': random.choice(list(self.device_types.keys()))
            }
            users.append(user)
        
        return pd.DataFrame(users)

    def generate_transactions(self, num_transactions: int, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic transaction data with fraud patterns."""
        transactions = []
        
        # Create user transaction history for velocity fraud detection
        user_transaction_history = {user_id: [] for user_id in users_df['user_id']}
        
        for i in range(num_transactions):
            # Select a user
            user = users_df.iloc[random.randint(0, len(users_df) - 1)]
            user_id = user['user_id']
            
            # Generate transaction timestamp
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Select merchant category
            category = random.choice(list(self.merchant_categories.keys()))
            category_info = self.merchant_categories[category]
            
            # Generate transaction amount based on category and user pattern
            base_amount = random.uniform(category_info['min_amount'], category_info['max_amount'])
            # Add some variation based on user pattern
            amount = base_amount * random.uniform(0.7, 1.3)
            
            # Select location (sometimes user's location, sometimes different)
            if random.random() < 0.8:  # 80% of transactions in user's location
                location = user['location']
            else:
                location = random.choice(list(self.locations.keys()))
            
            # Select device type
            if random.random() < 0.7:  # 70% of transactions on preferred device
                device_type = user['device_preference']
            else:
                device_type = random.choice(list(self.device_types.keys()))
            
            # Determine if this is fraud based on various patterns
            is_fraud = False
            fraud_reason = None
            
            # Check velocity fraud (recent transactions)
            recent_transactions = [t for t in user_transaction_history[user_id] 
                                 if (timestamp - t['timestamp']).total_seconds() < 3600]  # Last hour
            if len(recent_transactions) > 5 and random.random() < 0.2:
                is_fraud = True
                fraud_reason = 'velocity_fraud'
            
            # Check time anomaly
            hour = timestamp.hour
            if (hour < 6 or hour > 23) and random.random() < 0.1:
                is_fraud = True
                fraud_reason = 'time_anomaly'
            
            # Check merchant risk
            if category_info['fraud_rate'] > 0.10 and random.random() < 0.18:
                is_fraud = True
                fraud_reason = 'merchant_risk'
            
            # Check unusual amount
            if amount > user['avg_amount'] * 5 and random.random() < 0.15:
                is_fraud = True
                fraud_reason = 'unusual_amount'
            
            # Check geographic anomaly
            if location != user['location'] and random.random() < 0.12:
                is_fraud = True
                fraud_reason = 'geographic_anomaly'
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN_{str(uuid.uuid4())[:12].upper()}",
                'user_id': user_id,
                'amount': round(amount, 2),
                'merchant_category': category,
                'location': location,
                'device_type': device_type,
                'timestamp': timestamp,
                'is_fraud': int(is_fraud),
                'fraud_reason': fraud_reason,
                'user_pattern': user['pattern'],
                'user_avg_amount': user['avg_amount'],
                'user_avg_daily_transactions': user['avg_daily_transactions']
            }
            
            transactions.append(transaction)
            user_transaction_history[user_id].append(transaction)
        
        return pd.DataFrame(transactions)

    def generate_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features for ML model training."""
        df = transactions_df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log(df['amount'] + 1)
        df['amount_relative_to_avg'] = df['amount'] / df['user_avg_amount']
        df['is_high_amount'] = (df['amount'] > df['user_avg_amount'] * 3).astype(int)
        
        # Merchant category features
        category_fraud_rates = {cat: info['fraud_rate'] for cat, info in self.merchant_categories.items()}
        df['merchant_fraud_rate'] = df['merchant_category'].map(category_fraud_rates)
        
        # User pattern features
        pattern_fraud_rates = {pattern: info['fraud_rate'] for pattern, info in self.user_patterns.items()}
        df['user_pattern_fraud_rate'] = df['user_pattern'].map(pattern_fraud_rates)
        
        # Categorical encoding
        df = pd.get_dummies(df, columns=['merchant_category', 'device_type', 'user_pattern'], prefix=['cat', 'device', 'pattern'])
        
        return df

    def generate_dataset(self, num_users: int = 10000, num_transactions: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with users and transactions."""
        logger.info(f"Generating {num_users} users...")
        users_df = self.generate_user_profiles(num_users)
        
        logger.info(f"Generating {num_transactions} transactions...")
        transactions_df = self.generate_transactions(num_transactions, users_df)
        
        logger.info("Generating features...")
        features_df = self.generate_features(transactions_df)
        
        # Calculate fraud statistics
        fraud_rate = transactions_df['is_fraud'].mean()
        logger.info(f"Generated dataset with {len(features_df)} transactions and {fraud_rate:.2%} fraud rate")
        
        return users_df, transactions_df, features_df

    def save_dataset(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                    output_dir: str = "data") -> None:
        """Save the generated dataset to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        users_df.to_csv(f"{output_dir}/users.csv", index=False)
        transactions_df.to_csv(f"{output_dir}/transactions.csv", index=False)
        
        # Save as JSON for API testing
        transactions_df.head(1000).to_json(f"{output_dir}/sample_transactions.json", orient='records', date_format='iso')
        
        # Save dataset statistics
        stats = {
            'total_users': len(users_df),
            'total_transactions': len(transactions_df),
            'fraud_rate': transactions_df['is_fraud'].mean(),
            'fraud_reasons': transactions_df['fraud_reason'].value_counts().to_dict(),
            'merchant_categories': transactions_df['merchant_category'].value_counts().to_dict(),
            'locations': transactions_df['location'].value_counts().to_dict()
        }
        
        with open(f"{output_dir}/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}/")
        logger.info(f"Fraud rate: {stats['fraud_rate']:.2%}")
        logger.info(f"Total transactions: {stats['total_transactions']:,}")

def main():
    """Main function to generate the dataset."""
    # Initialize generator
    generator = FraudDataGenerator(seed=42)
    
    # Generate dataset
    users_df, transactions_df, features_df = generator.generate_dataset(
        num_users=10000,
        num_transactions=100000
    )
    
    # Save dataset
    generator.save_dataset(users_df, transactions_df, "data")
    
    # Print sample data
    print("\nSample transactions:")
    print(transactions_df[['transaction_id', 'user_id', 'amount', 'merchant_category', 'is_fraud']].head())
    
    print("\nFraud statistics:")
    print(transactions_df['fraud_reason'].value_counts())

if __name__ == "__main__":
    main() 
