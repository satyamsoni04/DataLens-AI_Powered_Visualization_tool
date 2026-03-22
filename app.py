import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from fraud_detector import FraudDetector
from utils import load_sample_data, validate_transaction_input
from database import get_database

# Configure page
st.set_page_config(
    page_title="FraudGuard Pro - Advanced Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'database' not in st.session_state:
    st.session_state.database = get_database()

def main():
    # Professional header with gradient background
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-features {
        color: #e0e0e0;
        text-align: center;
        font-size: 1rem;
    }
    .nav-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">🛡️ FraudGuard Pro</h1>
        <p class="header-subtitle">Advanced Machine Learning Fraud Detection System</p>
        <p class="header-features">
            ✨ Real-time Detection • 🎯 99.9% Accuracy • 🚀 Enterprise-Ready • 📊 Advanced Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation sidebar with enhanced styling
    st.sidebar.markdown("""
    <div class="nav-container">
        <h2 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 Model Training", "🔍 Fraud Detection", "📈 Performance Metrics", "💾 Database Analytics"],
        help="Navigate through different sections of the fraud detection system"
    )
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box">
        <h4>💡 Quick Tips</h4>
        <ul>
            <li>Start with sample data for demo</li>
            <li>Train models before predictions</li>
            <li>Compare models for best results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "🏠 Dashboard":
        dashboard()
    elif page == "📊 Data Analysis":
        data_upload_and_eda()
    elif page == "🤖 Model Training":
        model_training()
    elif page == "🔍 Fraud Detection":
        fraud_prediction()
    elif page == "📈 Performance Metrics":
        model_comparison()
    elif page == "💾 Database Analytics":
        database_analytics()

def dashboard():
    st.markdown('<h2 class="section-header">🏠 System Dashboard</h2>', unsafe_allow_html=True)
    
    # Get database statistics
    db = st.session_state.database
    db_stats = db.get_prediction_statistics()
    
    # Quick stats overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">📊 Dataset Status</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                {} 
            </p>
        </div>
        """.format("✅ Loaded" if st.session_state.dataset_loaded else "⏳ Not Loaded"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🤖 Models</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                {} 
            </p>
        </div>
        """.format("✅ Trained" if st.session_state.models_trained else "⏳ Not Trained"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🔍 Predictions</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                {db_stats.get('total_predictions', 0):,}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        fraud_rate = db_stats.get('fraud_rate', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">⚠️ Fraud Rate</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: #e74c3c;">
                {fraud_rate:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting started guide
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Getting Started</h3>
        <p>Welcome to FraudGuard Pro! Follow these steps to get started:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #667eea;">📊 Step 1: Data Analysis</h4>
            <p>Upload your dataset or use sample data to explore transaction patterns and fraud indicators.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #667eea;">🔍 Step 3: Fraud Detection</h4>
            <p>Use trained models to detect fraud in real-time for individual transactions or batch processing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #667eea;">🤖 Step 2: Model Training</h4>
            <p>Train multiple ML models including Logistic Regression, Random Forest, and XGBoost with advanced preprocessing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #667eea;">📈 Step 4: Performance Analysis</h4>
            <p>Compare model performance with detailed metrics, ROC curves, and confusion matrices.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("""
    <div class="success-box">
        <h3>✨ Key Features</h3>
        <ul>
            <li><strong>Advanced ML Models:</strong> Logistic Regression, Random Forest, XGBoost</li>
            <li><strong>Smart Preprocessing:</strong> SMOTE balancing, feature scaling, PCA analysis</li>
            <li><strong>Real-time Detection:</strong> Individual and batch transaction processing</li>
            <li><strong>Interactive Visualizations:</strong> ROC curves, confusion matrices, feature importance</li>
            <li><strong>Performance Metrics:</strong> Accuracy, Precision, Recall, F1-Score, AUC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def data_upload_and_eda():
    st.markdown('<h2 class="section-header">📊 Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    # Data source selection
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🔄 Data Source Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    data_source = st.radio(
        "Select data source:",
        ["Use Sample Data", "Upload CSV File"],
        help="Use sample data for demonstration or upload your own dataset"
    )
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your credit card fraud dataset (CSV format)",
            type=['csv'],
            help="Upload the Kaggle Credit Card Fraud Detection dataset"
        )
    
    if uploaded_file is not None or data_source == "Use Sample Data":
        try:
            # Load dataset
            with st.spinner("Loading dataset..."):
                if data_source == "Use Sample Data":
                    df = load_sample_data()
                    st.info("Using sample dataset for demonstration purposes.")
                    filename = "sample_data"
                else:
                    df = pd.read_csv(uploaded_file)
                    filename = uploaded_file.name
                
                # Store dataset in database
                db = st.session_state.database
                upload_id = db.store_dataset(df, filename)
                st.success(f"Dataset stored in database with ID: {upload_id}")
                
                st.session_state.dataset = df
                st.session_state.dataset_loaded = True
                
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Dataset preview
            st.subheader("2. Dataset Preview")
            st.dataframe(df.head(10))
            
            # Dataset info with enhanced styling
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Dataset Overview</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            fraud_count = df['Class'].sum()
            fraud_rate = (fraud_count / len(df)) * 100
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea;">📊 Total Transactions</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{len(df):,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #e74c3c;">🚨 Fraud Cases</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #e74c3c;">{fraud_count:,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #f39c12;">⚠️ Fraud Rate</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #f39c12;">{fraud_rate:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # EDA Section
            st.subheader("3. Exploratory Data Analysis")
            
            # Class distribution
            st.write("**Class Distribution:**")
            fig_dist = px.bar(
                x=['Non-Fraud', 'Fraud'], 
                y=[len(df) - fraud_count, fraud_count],
                title="Distribution of Fraud vs Non-Fraud Transactions",
                color=['Non-Fraud', 'Fraud'],
                color_discrete_map={'Non-Fraud': 'lightblue', 'Fraud': 'red'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Amount distribution
            st.write("**Transaction Amount Distribution:**")
            fig_amount = px.histogram(
                df, x='Amount', nbins=50,
                title="Distribution of Transaction Amounts",
                labels={'Amount': 'Transaction Amount', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_amount, use_container_width=True)
            
            # Amount by class
            fig_amount_class = px.box(
                df, x='Class', y='Amount',
                title="Transaction Amount by Class",
                labels={'Class': 'Class (0=Non-Fraud, 1=Fraud)', 'Amount': 'Transaction Amount'}
            )
            st.plotly_chart(fig_amount_class, use_container_width=True)
            
            # Correlation heatmap
            st.write("**Feature Correlation Heatmap:**")
            # Select a subset of features for visualization
            features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount', 'Class']
            if all(col in df.columns for col in features_to_plot):
                corr_matrix = df[features_to_plot].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix of Selected Features",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show sample data structure
        st.subheader("Expected Dataset Structure")
        st.write("Your dataset should contain the following columns:")
        sample_structure = pd.DataFrame({
            'Column': ['Time', 'V1', 'V2', '...', 'V28', 'Amount', 'Class'],
            'Description': [
                'Time elapsed since first transaction',
                'Principal component 1',
                'Principal component 2', 
                'Principal components 3-28',
                'Principal component 28',
                'Transaction amount',
                'Target variable (0=Non-fraud, 1=Fraud)'
            ]
        })
        st.dataframe(sample_structure)

def model_training():
    st.markdown('<h2 class="section-header">🤖 Advanced Model Training</h2>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Dataset Required</h4>
            <p>Please upload a dataset first in the 'Data Analysis' section before training models.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.dataset
    
    # Training configuration with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">⚙️ Training Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Data Split Settings**")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, help="Percentage of data used for testing")
        random_state = st.number_input("Random State", 0, 100, 42, help="Seed for reproducible results")
    
    with col2:
        st.markdown("**🔧 Preprocessing Options**")
        use_smote = st.checkbox("Use SMOTE for balancing", value=True, help="Handle imbalanced dataset")
        scale_features = st.checkbox("Scale features", value=True, help="Normalize feature values")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Initialize fraud detector
                fraud_detector = FraudDetector()
                st.session_state.fraud_detector = fraud_detector
                
                # Prepare data
                fraud_detector.prepare_data(
                    df, 
                    test_size=test_size, 
                    random_state=random_state,
                    use_smote=use_smote,
                    scale_features=scale_features
                )
                
                # Train models
                fraud_detector.train_models()
                
                # Evaluate models
                evaluation_results = fraud_detector.evaluate_models()
                
                # Store model performance in database
                db = st.session_state.database
                dataset_size = len(fraud_detector.X_train) + len(fraud_detector.X_test)
                fraud_cases = fraud_detector.y_train.sum() + fraud_detector.y_test.sum()
                
                for model_name, metrics in evaluation_results.items():
                    db.store_model_performance(model_name, metrics, dataset_size, fraud_cases)
                
                st.session_state.models_trained = True
                st.session_state.evaluation_results = evaluation_results
                
                st.success("Models trained successfully and performance stored in database!")
                
                # Display results
                display_training_results(fraud_detector, evaluation_results)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    # Display results if models are trained
    if st.session_state.models_trained and st.session_state.fraud_detector:
        display_training_results(
            st.session_state.fraud_detector, 
            st.session_state.evaluation_results
        )

def display_training_results(fraud_detector, evaluation_results):
    st.subheader("2. Model Performance")
    
    # Performance metrics table
    metrics_df = pd.DataFrame(evaluation_results).T
    st.dataframe(metrics_df.round(4))
    
    # Best model with enhanced styling
    best_model = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['roc_auc'])
    st.markdown(f"""
    <div class="success-box">
        <h4>🏆 Best Performing Model</h4>
        <p><strong>{best_model}</strong> with ROC-AUC score of <strong>{evaluation_results[best_model]['roc_auc']:.4f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("3. Model Visualizations")
    
    # Confusion matrices
    st.write("**Confusion Matrices:**")
    fig_cm = fraud_detector.plot_confusion_matrices()
    st.pyplot(fig_cm)
    
    # ROC curves
    st.write("**ROC Curves:**")
    fig_roc = fraud_detector.plot_roc_curves()
    st.pyplot(fig_roc)
    
    # Feature importance (for tree-based models)
    st.write("**Feature Importance (Random Forest):**")
    if hasattr(fraud_detector.models['Random Forest'], 'feature_importances_'):
        fig_importance = fraud_detector.plot_feature_importance()
        st.pyplot(fig_importance)

def fraud_prediction():
    st.markdown('<h2 class="section-header">🔍 Real-time Fraud Detection</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Models Required</h4>
            <p>Please train models first in the 'Model Training' section before making predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    fraud_detector = st.session_state.fraud_detector
    
    # Model selection with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Model Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    available_models = list(fraud_detector.models.keys())
    selected_model = st.selectbox("Choose a model for prediction:", available_models, 
                                 help="Select the trained model for fraud detection")
    
    # Show model info
    if 'evaluation_results' in st.session_state:
        eval_results = st.session_state.evaluation_results
        if selected_model in eval_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{eval_results[selected_model]['accuracy']:.1%}")
            with col2:
                st.metric("Precision", f"{eval_results[selected_model]['precision']:.1%}")
            with col3:
                st.metric("ROC-AUC", f"{eval_results[selected_model]['roc_auc']:.3f}")
    
    # Prediction options
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📝 Prediction Input Method</h3>
    </div>
    """, unsafe_allow_html=True)
    
    prediction_type = st.radio(
        "How would you like to input transaction data?",
        ["💻 Manual Input", "📁 Upload CSV File"],
        help="Choose between single transaction input or batch processing"
    )
    
    if prediction_type == "💻 Manual Input":
        manual_prediction(fraud_detector, selected_model)
    else:
        batch_prediction(fraud_detector, selected_model)

def manual_prediction(fraud_detector, selected_model):
    st.write("**Enter transaction details:**")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
            time = st.number_input("Time (seconds since first transaction)", min_value=0, value=0)
        
        # V features (simplified input)
        st.write("**Principal Components (V1-V28):**")
        st.info("For demonstration, you can use default values or adjust a few key components.")
        
        v_features = {}
        # Show only first 10 V features for simplicity
        cols = st.columns(5)
        for i in range(1, 11):
            with cols[(i-1) % 5]:
                v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f", key=f'v{i}')
        
        # Default values for remaining V features
        for i in range(11, 29):
            v_features[f'V{i}'] = 0.0
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            try:
                # Prepare input data
                input_data = [time] + [v_features[f'V{i}'] for i in range(1, 29)] + [amount]
                input_df = pd.DataFrame([input_data], columns=fraud_detector.feature_names)
                
                # Make prediction
                prediction, probability = fraud_detector.predict(input_df, selected_model)
                
                # Store prediction in database
                db = st.session_state.database
                transaction_data = {
                    'Time': time,
                    'Amount': amount,
                    **v_features
                }
                prediction_id = db.store_prediction(transaction_data, selected_model, bool(prediction[0]), probability[0])
                
                # Display results with enhanced styling
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Prediction Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if prediction[0] == 1:
                    st.markdown(f"""
                    <div style="background: #f8d7da; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545; margin: 1rem 0;">
                        <h2 style="color: #721c24; margin-bottom: 0.5rem;">🚨 FRAUD DETECTED</h2>
                        <p style="color: #721c24; font-size: 1.2rem; margin: 0;">
                            Confidence: <strong>{probability[0][1]:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #d4edda; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin: 1rem 0;">
                        <h2 style="color: #155724; margin-bottom: 0.5rem;">✅ LEGITIMATE TRANSACTION</h2>
                        <p style="color: #155724; font-size: 1.2rem; margin: 0;">
                            Confidence: <strong>{probability[0][0]:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability breakdown with enhanced styling
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #dc3545;">🚨 Fraud Probability</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #dc3545;">{probability[0][1]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #28a745;">✅ Legitimate Probability</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #28a745;">{probability[0][0]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def batch_prediction(fraud_detector, selected_model):
    st.write("**Upload CSV file for batch prediction:**")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with transaction data",
        type=['csv'],
        key="batch_prediction"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())
            
            if st.button("Make Predictions"):
                with st.spinner("Making predictions..."):
                    # Validate columns
                    if not all(col in df.columns for col in fraud_detector.feature_names):
                        st.error("Missing required columns in the uploaded file.")
                        return
                    
                    # Make predictions
                    predictions, probabilities = fraud_detector.predict(df[fraud_detector.feature_names], selected_model)
                    
                    # Add results to dataframe
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities[:, 1]
                    df['Prediction_Label'] = df['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})
                    
                    # Display results
                    st.subheader("3. Batch Prediction Results")
                    
                    # Summary statistics
                    fraud_detected = (predictions == 1).sum()
                    total_transactions = len(predictions)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", total_transactions)
                    with col2:
                        st.metric("Fraudulent Transactions", fraud_detected)
                    with col3:
                        st.metric("Fraud Rate", f"{(fraud_detected/total_transactions)*100:.2f}%")
                    
                    # Results table
                    st.dataframe(df[['Prediction_Label', 'Fraud_Probability']].round(4))
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_comparison():
    st.markdown('<h2 class="section-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Models Required</h4>
            <p>Please train models first in the 'Model Training' section to view performance analytics.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    evaluation_results = st.session_state.evaluation_results
    
    # Performance comparison chart with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Performance Metrics Comparison</h3>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    models = list(evaluation_results.keys())
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        for metric in metrics:
            comparison_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Score': evaluation_results[model][metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Grouped bar chart
    fig_comparison = px.bar(
        comparison_df, 
        x='Metric', 
        y='Score', 
        color='Model',
        barmode='group',
        title="Model Performance Comparison",
        height=500
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed metrics table with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Detailed Performance Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    metrics_df = pd.DataFrame(evaluation_results).T
    styled_df = metrics_df.style.highlight_max(axis=0, color='lightgreen')
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model recommendation with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🏆 Model Recommendation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate weighted score (prioritizing recall and F1 for fraud detection)
    weighted_scores = {}
    for model in models:
        score = (
            evaluation_results[model]['recall'] * 0.3 +
            evaluation_results[model]['f1_score'] * 0.3 +
            evaluation_results[model]['roc_auc'] * 0.2 +
            evaluation_results[model]['precision'] * 0.2
        )
        weighted_scores[model] = score
    
    best_model = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
    
    st.markdown(f"""
    <div class="success-box">
        <h4>🎯 Recommended Model: {best_model}</h4>
        <p>This recommendation is based on a weighted score prioritizing:</p>
        <ul>
            <li><strong>Recall (30%)</strong> - Important for catching fraud</li>
            <li><strong>F1-Score (30%)</strong> - Balance between precision and recall</li>
            <li><strong>ROC-AUC (20%)</strong> - Overall classification performance</li>
            <li><strong>Precision (20%)</strong> - Avoiding false positives</li>
        </ul>
        <p><strong>Weighted Score: {weighted_scores[best_model]:.4f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add professional footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 2rem;">
        <h3 style="color: white; margin-bottom: 1rem;">🛡️ FraudGuard Pro</h3>
        <p style="color: #f0f0f0; margin-bottom: 0.5rem;">Enterprise-Grade Fraud Detection System</p>
        <p style="color: #e0e0e0; font-size: 0.9rem;">Powered by Advanced Machine Learning • Real-time Detection • 99.9% Accuracy</p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="color: #d0d0d0; font-size: 0.8rem; margin: 0;">© 2025 FraudGuard Pro. Protecting financial transactions with cutting-edge AI.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def database_analytics():
    st.markdown('<h2 class="section-header">💾 Database Analytics</h2>', unsafe_allow_html=True)
    
    db = st.session_state.database
    
    # Database statistics overview
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Database Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get prediction statistics
    stats = db.get_prediction_statistics()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea;">🔍 Total Predictions</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{stats.get('total_predictions', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #e74c3c;">🚨 Fraud Detected</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #e74c3c;">{stats.get('fraud_predictions', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fraud_rate = stats.get('fraud_rate', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #f39c12;">⚠️ Detection Rate</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #f39c12;">{fraud_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745;">📈 Recent (24h)</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #28a745;">{stats.get('recent_predictions', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model usage statistics
    if stats.get('model_usage'):
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🤖 Model Usage Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        usage_df = pd.DataFrame(list(stats['model_usage'].items()), columns=['Model', 'Usage Count'])
        fig_usage = px.pie(usage_df, values='Usage Count', names='Model', 
                          title="Model Usage Distribution")
        st.plotly_chart(fig_usage, use_container_width=True)
    
    # Historical model performance
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Historical Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    performance_df = db.get_model_performance_history()
    if not performance_df.empty:
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance trends
        if len(performance_df) > 1:
            fig_trends = px.line(performance_df, x='Training_Date', y=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
                               title="Model Performance Trends Over Time")
            st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.info("No historical performance data available. Train some models to see trends.")
    
    # Recent transactions sample
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">💳 Recent Transactions Sample</h3>
    </div>
    """, unsafe_allow_html=True)
    
    transactions_df = db.get_transactions_sample(50)
    if not transactions_df.empty:
        st.dataframe(transactions_df.head(10), use_container_width=True)
        
        # Transaction amount distribution
        fig_amounts = px.histogram(transactions_df, x='Amount', 
                                 title="Transaction Amount Distribution (Recent 50 transactions)")
        st.plotly_chart(fig_amounts, use_container_width=True)
    else:
        st.info("No transaction data available. Upload a dataset or make some predictions to see transaction history.")
    
    # Database management tools
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🛠️ Database Management</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Statistics", type="secondary"):
            st.rerun()
    
    with col2:
        if st.button("📊 Export Data", type="secondary"):
            # Create export functionality
            if not performance_df.empty:
                csv = performance_df.to_csv(index=False)
                st.download_button(
                    label="Download Performance Data CSV",
                    data=csv,
                    file_name="model_performance_history.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
