import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import Dict, Tuple
import streamlit as st

class FraudDatabase:
    def __init__(self):
        """Initialize SQLite database for cloud deployment"""
        self.db_path = "fraud_detection.db"
        self.connection = None
        self.connect()
        self.init_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            print("✅ SQLite database connected successfully")
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            raise
    
    def init_tables(self):
        """Initialize database tables"""
        cursor = self.connection.cursor()
        
        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_transactions INTEGER,
                fraud_cases INTEGER,
                fraud_percentage REAL
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                dataset_size INTEGER,
                fraud_cases INTEGER
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                transaction_time REAL,
                transaction_amount REAL,
                prediction_result INTEGER,
                fraud_probability REAL,
                normal_probability REAL
            )
        """)
        
        # Transactions table (simplified for SQLite)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_feature REAL,
                amount REAL,
                class_label INTEGER
            )
        """)
        
        self.connection.commit()
        print("✅ Database tables initialized")
    
    def store_dataset(self, df: pd.DataFrame, filename: str = "sample_data") -> int:
        """Store dataset in database"""
        cursor = self.connection.cursor()
        
        total_transactions = len(df)
        fraud_cases = df['Class'].sum() if 'Class' in df.columns else 0
        fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
        
        cursor.execute("""
            INSERT INTO datasets (filename, total_transactions, fraud_cases, fraud_percentage)
            VALUES (?, ?, ?, ?)
        """, (filename, total_transactions, fraud_cases, fraud_percentage))
        
        dataset_id = cursor.lastrowid
        
        # Store sample transactions
        for _, row in df.head(1000).iterrows():  # Limit to 1000 for performance
            cursor.execute("""
                INSERT INTO transactions (time_feature, amount, class_label)
                VALUES (?, ?, ?)
            """, (row.get('Time', 0), row.get('Amount', 0), row.get('Class', 0)))
        
        self.connection.commit()
        return dataset_id
    
    def store_model_performance(self, model_name: str, metrics: Dict, dataset_size: int, fraud_cases: int):
        """Store model performance metrics"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_name, accuracy, precision, recall, f1_score, roc_auc, dataset_size, fraud_cases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('roc_auc', 0),
            dataset_size,
            fraud_cases
        ))
        
        self.connection.commit()
    
    def store_prediction(self, transaction_data: Dict, model_name: str, prediction: bool, probabilities: Tuple[float, float]) -> int:
        """Store individual prediction result"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (model_name, transaction_time, transaction_amount, prediction_result, fraud_probability, normal_probability)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            transaction_data.get('Time', 0),
            transaction_data.get('Amount', 0),
            int(prediction),
            probabilities[1] if len(probabilities) > 1 else probabilities[0],
            probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
        ))
        
        prediction_id = cursor.lastrowid
        self.connection.commit()
        return prediction_id
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Get historical model performance data"""
        return pd.read_sql_query("""
            SELECT model_name as Model, training_date as Training_Date, 
                   accuracy as Accuracy, precision as Precision, recall as Recall,
                   f1_score as F1_Score, roc_auc as ROC_AUC, dataset_size as Dataset_Size
            FROM model_performance 
            ORDER BY training_date DESC
        """, self.connection)
    
    def get_prediction_statistics(self) -> Dict:
        """Get prediction statistics"""
        cursor = self.connection.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Fraud predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_result = 1")
        fraud_predictions = cursor.fetchone()[0]
        
        # Recent predictions (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE prediction_date >= datetime('now', '-1 day')
        """)
        recent_predictions = cursor.fetchone()[0]
        
        # Model usage
        cursor.execute("""
            SELECT model_name, COUNT(*) as usage_count 
            FROM predictions 
            GROUP BY model_name
        """)
        model_usage = dict(cursor.fetchall())
        
        fraud_rate = (fraud_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'fraud_predictions': fraud_predictions,
            'fraud_rate': fraud_rate,
            'recent_predictions': recent_predictions,
            'model_usage': model_usage
        }
    
    def get_transactions_sample(self, limit: int = 100) -> pd.DataFrame:
        """Get sample transactions from database"""
        return pd.read_sql_query(f"""
            SELECT time_feature as Time, amount as Amount, class_label as Class
            FROM transactions 
            ORDER BY upload_date DESC 
            LIMIT {limit}
        """, self.connection)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

@st.cache_resource
def get_database():
    """Get cached database connection"""
    return FraudDatabase()