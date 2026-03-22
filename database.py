import psycopg2
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st

class FraudDatabase:
    def __init__(self):
        self.connection = None
        self.connect()
        self.init_tables()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            # Use DATABASE_URL if available, otherwise individual parameters
            if 'DATABASE_URL' in os.environ:
                self.connection = psycopg2.connect(os.environ['DATABASE_URL'])
            else:
                self.connection = psycopg2.connect(
                    host=os.environ.get('PGHOST', 'localhost'),
                    database=os.environ.get('PGDATABASE', 'fraud_detection'),
                    user=os.environ.get('PGUSER', 'postgres'),
                    password=os.environ.get('PGPASSWORD', ''),
                    port=os.environ.get('PGPORT', '5432')
                )
            self.connection.autocommit = True
            print("✅ Database connected successfully")
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            raise
    
    def init_tables(self):
        """Initialize database tables"""
        cursor = self.connection.cursor()
        
        # Transactions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            time_seconds FLOAT,
            v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT,
            v6 FLOAT, v7 FLOAT, v8 FLOAT, v9 FLOAT, v10 FLOAT,
            v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT, v15 FLOAT,
            v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
            v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT,
            v26 FLOAT, v27 FLOAT, v28 FLOAT,
            amount FLOAT,
            is_fraud BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Model performance table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            accuracy FLOAT,
            precision_score FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dataset_size INTEGER,
            fraud_cases INTEGER
        )
        """)
        
        # Predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            transaction_id INTEGER REFERENCES transactions(id),
            model_name VARCHAR(100),
            prediction BOOLEAN,
            fraud_probability FLOAT,
            legitimate_probability FLOAT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Dataset uploads table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_uploads (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            total_transactions INTEGER,
            fraud_cases INTEGER,
            fraud_rate FLOAT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.close()
        print("✅ Database tables initialized")
    
    def store_dataset(self, df: pd.DataFrame, filename: str = "sample_data") -> int:
        """Store dataset in database"""
        cursor = self.connection.cursor()
        
        try:
            # Store dataset metadata
            total_transactions = len(df)
            fraud_cases = df['Class'].sum()
            fraud_rate = (fraud_cases / total_transactions) * 100
            
            cursor.execute("""
            INSERT INTO dataset_uploads (filename, total_transactions, fraud_cases, fraud_rate)
            VALUES (%s, %s, %s, %s) RETURNING id
            """, (filename, total_transactions, fraud_cases, fraud_rate))
            
            upload_id = cursor.fetchone()[0]
            
            # Store individual transactions
            for _, row in df.iterrows():
                v_values = [row[f'V{i}'] for i in range(1, 29)]
                cursor.execute("""
                INSERT INTO transactions (
                    time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                    v21, v22, v23, v24, v25, v26, v27, v28, amount, is_fraud
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [row['Time']] + v_values + [row['Amount'], bool(row['Class'])])
            
            cursor.close()
            print(f"✅ Dataset stored: {total_transactions} transactions, {fraud_cases} fraud cases")
            return upload_id
            
        except Exception as e:
            cursor.close()
            print(f"❌ Error storing dataset: {str(e)}")
            raise
    
    def store_model_performance(self, model_name: str, metrics: Dict, dataset_size: int, fraud_cases: int):
        """Store model performance metrics"""
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO model_performance (
                model_name, accuracy, precision_score, recall, f1_score, roc_auc,
                dataset_size, fraud_cases
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_name,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc'],
                dataset_size,
                fraud_cases
            ))
            
            cursor.close()
            print(f"✅ Model performance stored for {model_name}")
            
        except Exception as e:
            cursor.close()
            print(f"❌ Error storing model performance: {str(e)}")
            raise
    
    def store_prediction(self, transaction_data: Dict, model_name: str, prediction: bool, probabilities: Tuple[float, float]) -> int:
        """Store individual prediction result"""
        cursor = self.connection.cursor()
        
        try:
            # First store the transaction if it doesn't exist
            v_values = [transaction_data.get(f'V{i}', 0.0) for i in range(1, 29)]
            cursor.execute("""
            INSERT INTO transactions (
                time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                v21, v22, v23, v24, v25, v26, v27, v28, amount, is_fraud
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                     %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                     %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL) RETURNING id
            """, [transaction_data.get('Time', 0)] + v_values + [transaction_data.get('Amount', 0)])
            
            transaction_id = cursor.fetchone()[0]
            
            # Store the prediction
            cursor.execute("""
            INSERT INTO predictions (
                transaction_id, model_name, prediction, fraud_probability, legitimate_probability
            ) VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (transaction_id, model_name, prediction, probabilities[1], probabilities[0]))
            
            prediction_id = cursor.fetchone()[0]
            cursor.close()
            
            return prediction_id
            
        except Exception as e:
            cursor.close()
            print(f"❌ Error storing prediction: {str(e)}")
            raise
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Get historical model performance data"""
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("""
            SELECT model_name, accuracy, precision_score, recall, f1_score, roc_auc,
                   training_date, dataset_size, fraud_cases
            FROM model_performance
            ORDER BY training_date DESC
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            if results:
                df = pd.DataFrame(results, columns=[
                    'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC',
                    'Training_Date', 'Dataset_Size', 'Fraud_Cases'
                ])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            cursor.close()
            print(f"❌ Error retrieving model performance: {str(e)}")
            return pd.DataFrame()
    
    def get_prediction_statistics(self) -> Dict:
        """Get prediction statistics"""
        cursor = self.connection.cursor()
        
        try:
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Fraud predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = true")
            fraud_predictions = cursor.fetchone()[0]
            
            # Recent predictions (last 24 hours)
            cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE prediction_date >= NOW() - INTERVAL '24 hours'
            """)
            recent_predictions = cursor.fetchone()[0]
            
            # Model usage
            cursor.execute("""
            SELECT model_name, COUNT(*) as usage_count
            FROM predictions
            GROUP BY model_name
            ORDER BY usage_count DESC
            """)
            model_usage = cursor.fetchall()
            
            cursor.close()
            
            return {
                'total_predictions': total_predictions,
                'fraud_predictions': fraud_predictions,
                'fraud_rate': (fraud_predictions / total_predictions * 100) if total_predictions > 0 else 0,
                'recent_predictions': recent_predictions,
                'model_usage': dict(model_usage) if model_usage else {}
            }
            
        except Exception as e:
            cursor.close()
            print(f"❌ Error retrieving prediction statistics: {str(e)}")
            return {}
    
    def get_transactions_sample(self, limit: int = 100) -> pd.DataFrame:
        """Get sample transactions from database"""
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(f"""
            SELECT time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                   v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                   v21, v22, v23, v24, v25, v26, v27, v28, amount, is_fraud
            FROM transactions
            ORDER BY created_at DESC
            LIMIT {limit}
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            if results:
                columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
                df = pd.DataFrame(results, columns=columns)
                df['Class'] = df['Class'].astype(int)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            cursor.close()
            print(f"❌ Error retrieving transactions: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")

# Initialize database connection
@st.cache_resource
def get_database():
    """Get cached database connection"""
    return FraudDatabase()