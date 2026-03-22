import pandas as pd
import numpy as np

def load_sample_data():
    """Generate sample credit card fraud data for demonstration"""
    np.random.seed(42)
    
    # Generate sample data with the same structure as the Kaggle dataset
    n_samples = 1000
    n_normal = 950
    n_fraud = 50
    
    # Time feature
    time = np.random.randint(0, 172800, n_samples)  # 48 hours in seconds
    
    # V1-V28 features (PCA components)
    v_features = {}
    for i in range(1, 29):
        # Normal transactions
        v_normal = np.random.normal(0, 1, n_normal)
        # Fraud transactions (slightly different distribution)
        v_fraud = np.random.normal(0.5, 1.5, n_fraud)
        v_features[f'V{i}'] = np.concatenate([v_normal, v_fraud])
    
    # Amount feature
    amount_normal = np.random.lognormal(3, 1.5, n_normal)
    amount_fraud = np.random.lognormal(2, 2, n_fraud)
    amount = np.concatenate([amount_normal, amount_fraud])
    
    # Class (target)
    class_labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Create DataFrame
    data = {'Time': time}
    data.update(v_features)
    data.update({'Amount': amount, 'Class': class_labels})
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def validate_transaction_input(input_data):
    """Validate transaction input data"""
    required_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    if not isinstance(input_data, dict):
        return False, "Input data must be a dictionary"
    
    missing_features = set(required_features) - set(input_data.keys())
    if missing_features:
        return False, f"Missing features: {missing_features}"
    
    # Check data types
    try:
        for feature in required_features:
            float(input_data[feature])
    except (ValueError, TypeError):
        return False, f"All features must be numeric"
    
    # Check reasonable ranges
    if input_data['Amount'] < 0:
        return False, "Amount cannot be negative"
    
    if input_data['Time'] < 0:
        return False, "Time cannot be negative"
    
    return True, "Valid input"

def format_prediction_result(prediction, probability, model_name):
    """Format prediction results for display"""
    result = {
        'model': model_name,
        'prediction': 'Fraud' if prediction[0] == 1 else 'Legitimate',
        'fraud_probability': probability[0][1],
        'legitimate_probability': probability[0][0],
        'confidence': max(probability[0])
    }
    return result

def calculate_model_metrics_summary(evaluation_results):
    """Calculate summary statistics for model comparison"""
    if not evaluation_results:
        return {}
    
    models = list(evaluation_results.keys())
    metrics = list(evaluation_results[models[0]].keys())
    
    summary = {}
    for metric in metrics:
        values = [evaluation_results[model][metric] for model in models]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'best_model': models[np.argmax(values)]
        }
    
    return summary

def get_feature_statistics(df):
    """Get basic statistics for dataset features"""
    if df is None or df.empty:
        return {}
    
    stats = {
        'total_transactions': len(df),
        'fraud_cases': df['Class'].sum() if 'Class' in df.columns else 0,
        'fraud_rate': (df['Class'].sum() / len(df)) * 100 if 'Class' in df.columns else 0,
        'amount_stats': {
            'mean': df['Amount'].mean() if 'Amount' in df.columns else 0,
            'median': df['Amount'].median() if 'Amount' in df.columns else 0,
            'std': df['Amount'].std() if 'Amount' in df.columns else 0,
            'min': df['Amount'].min() if 'Amount' in df.columns else 0,
            'max': df['Amount'].max() if 'Amount' in df.columns else 0
        }
    }
    
    return stats

def preprocess_uploaded_data(df):
    """Preprocess uploaded data to ensure compatibility"""
    try:
        # Check required columns
        required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        
        # Remove any extra columns
        available_columns = [col for col in required_columns if col in df.columns]
        df_processed = df[available_columns].copy()
        
        # Handle missing values
        df_processed = df_processed.dropna()
        
        # Ensure proper data types
        for col in df_processed.columns:
            if col != 'Class':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            else:
                df_processed[col] = df_processed[col].astype(int)
        
        # Remove rows with invalid data
        df_processed = df_processed.dropna()
        
        # Validate Class column values
        if 'Class' in df_processed.columns:
            valid_classes = df_processed['Class'].isin([0, 1])
            df_processed = df_processed[valid_classes]
        
        return df_processed, True, "Data preprocessed successfully"
        
    except Exception as e:
        return df, False, f"Error preprocessing data: {str(e)}"

def generate_model_performance_report(evaluation_results):
    """Generate a detailed performance report"""
    if not evaluation_results:
        return "No evaluation results available."
    
    report = "# Model Performance Report\n\n"
    
    for model_name, metrics in evaluation_results.items():
        report += f"## {model_name}\n"
        report += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
        report += f"- **Precision**: {metrics['precision']:.4f}\n"
        report += f"- **Recall**: {metrics['recall']:.4f}\n"
        report += f"- **F1-Score**: {metrics['f1_score']:.4f}\n"
        report += f"- **ROC-AUC**: {metrics['roc_auc']:.4f}\n\n"
    
    # Find best model for each metric
    report += "## Best Models by Metric\n"
    metrics_list = list(next(iter(evaluation_results.values())).keys())
    
    for metric in metrics_list:
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x][metric])
        best_score = evaluation_results[best_model][metric]
        report += f"- **{metric.replace('_', ' ').title()}**: {best_model} ({best_score:.4f})\n"
    
    return report
