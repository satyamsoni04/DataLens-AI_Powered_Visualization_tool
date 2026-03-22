import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class FraudDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_fitted = False
        
    def prepare_data(self, df, test_size=0.2, random_state=42, use_smote=True, scale_features=True):
        """Prepare data for training"""
        try:
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class']
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            if scale_features:
                self.X_train = pd.DataFrame(
                    self.scaler.fit_transform(self.X_train),
                    columns=self.feature_names
                )
                self.X_test = pd.DataFrame(
                    self.scaler.transform(self.X_test),
                    columns=self.feature_names
                )
            
            # Apply SMOTE for balancing
            if use_smote:
                smote = SMOTE(random_state=random_state)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"Data prepared successfully!")
            print(f"Training set size: {len(self.X_train)}")
            print(f"Test set size: {len(self.X_test)}")
            print(f"Fraud cases in training: {sum(self.y_train)}")
            
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")
    
    def train_models(self):
        """Train all models"""
        if self.X_train is None:
            raise Exception("Data not prepared. Call prepare_data() first.")
        
        try:
            # Initialize models
            models_config = {
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'XGBoost': xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            }
            
            # Train each model
            for name, model in models_config.items():
                print(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                
                # Save model
                model_path = f"model_{name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model, model_path)
                print(f"{name} saved to {model_path}")
            
            self.is_fitted = True
            print("All models trained successfully!")
            
        except Exception as e:
            raise Exception(f"Error training models: {str(e)}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if not self.is_fitted:
            raise Exception("Models not trained. Call train_models() first.")
        
        evaluation_results = {}
        
        try:
            for name, model in self.models.items():
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                results = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                evaluation_results[name] = results
                print(f"{name} evaluation completed.")
            
            return evaluation_results
            
        except Exception as e:
            raise Exception(f"Error evaluating models: {str(e)}")
    
    def predict(self, X, model_name):
        """Make predictions using specified model"""
        if not self.is_fitted:
            raise Exception("Models not trained.")
        
        if model_name not in self.models:
            raise Exception(f"Model {model_name} not found.")
        
        try:
            # Scale features if scaler was fitted
            if hasattr(self.scaler, 'mean_'):
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=self.feature_names
                )
            else:
                X_scaled = X
            
            model = self.models[model_name]
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[idx],
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud']
            )
            axes[idx].set_title(f'{name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' not in self.models:
            raise Exception("Random Forest model not found.")
        
        model = self.models['Random Forest']
        if not hasattr(model, 'feature_importances_'):
            raise Exception("Model does not have feature importance.")
        
        # Get top 20 features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=feature_importance,
            x='importance',
            y='feature',
            ax=ax
        )
        ax.set_title('Top 20 Feature Importance (Random Forest)')
        ax.set_xlabel('Importance')
        
        return fig
    
    def save_models(self, directory='models'):
        """Save all trained models"""
        if not self.is_fitted:
            raise Exception("Models not trained.")
        
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(directory, filename)
            joblib.dump(model, filepath)
        
        # Save scaler
        scaler_path = os.path.join(directory, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Models saved to {directory} directory.")
    
    def load_models(self, directory='models'):
        """Load trained models"""
        try:
            # Load models
            model_files = {
                'Logistic Regression': 'logistic_regression.pkl',
                'Random Forest': 'random_forest.pkl',
                'XGBoost': 'xgboost.pkl'
            }
            
            for name, filename in model_files.items():
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    self.models[name] = joblib.load(filepath)
            
            # Load scaler
            scaler_path = os.path.join(directory, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.is_fitted = len(self.models) > 0
            print(f"Loaded {len(self.models)} models from {directory}.")
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
