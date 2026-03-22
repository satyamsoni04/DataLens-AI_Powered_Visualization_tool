import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fraud_detector import FraudDetector
from utils import *
import os

# Try PostgreSQL first (for Replit), fallback to SQLite (for Streamlit Cloud)
try:
    if 'DATABASE_URL' in os.environ or all(key in os.environ for key in ['PGHOST', 'PGDATABASE', 'PGUSER', 'PGPASSWORD', 'PGPORT']):
        from database import get_database
        DATABASE_TYPE = "PostgreSQL"
    else:
        from database_cloud import get_database
        DATABASE_TYPE = "SQLite"
except ImportError:
    from database_cloud import get_database
    DATABASE_TYPE = "SQLite"

# Page configuration
st.set_page_config(
    page_title="FraudGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .section-header {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'fraud_detector' not in st.session_state:
        st.session_state.fraud_detector = FraudDetector()
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    if 'database' not in st.session_state:
        st.session_state.database = get_database()
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1>🛡️ FraudGuard Pro</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Enterprise-Grade Credit Card Fraud Detection System</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Powered by Advanced Machine Learning • {DATABASE_TYPE} Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 Model Training", "🔍 Fraud Detection", "📈 Performance Metrics", "💾 Database Analytics"],
        help="Navigate through different sections of the fraud detection system"
    )
    
    # Quick stats in sidebar
    st.sidebar.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <h4 style="color: #667eea; margin-bottom: 0.5rem;">🚀 Quick Access</h4>
        <ul style="padding-left: 1rem; margin: 0;">
            <li>Upload your dataset</li>
            <li>Train ML models</li>
            <li>Detect fraud in real-time</li>
            <li>Analyze performance metrics</li>
            <li>View database insights</li>
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
    stats = db.get_prediction_statistics()
    
    # Key metrics
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
        models_status = "✅ Ready" if st.session_state.models_trained else "⏳ Not Trained"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745;">🤖 Models Status</h4>
            <p style="font-size: 1.2rem; font-weight: bold; color: #28a745;">{models_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System overview
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 10px; margin: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 System Overview</h3>
        <p>FraudGuard Pro is an enterprise-grade fraud detection system featuring:</p>
        <ul>
            <li><strong>Multiple ML Models:</strong> Logistic Regression, Random Forest, XGBoost</li>
            <li><strong>Real-time Detection:</strong> Instant fraud analysis for transactions</li>
            <li><strong>Data Persistence:</strong> Complete audit trail with database storage</li>
            <li><strong>Performance Analytics:</strong> Comprehensive model evaluation metrics</li>
            <li><strong>Professional Interface:</strong> Intuitive design for business users</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def data_upload_and_eda():
    st.markdown('<h2 class="section-header">📊 Data Analysis & Upload</h2>', unsafe_allow_html=True)
    
    # Data upload section
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📁 Data Upload</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your fraud detection dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file with transaction data including features V1-V28, Time, Amount, and Class columns"
        )
    
    with col2:
        use_sample = st.button("🎲 Use Sample Data", type="primary")
    
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_uploaded_data(df)
        st.session_state.data = df
        
        # Store in database
        db = st.session_state.database
        dataset_id = db.store_dataset(df, uploaded_file.name)
        st.success(f"✅ Dataset uploaded successfully! Database ID: {dataset_id}")
        
    elif use_sample:
        df = load_sample_data()
        st.session_state.data = df
        
        # Store in database
        db = st.session_state.database
        dataset_id = db.store_dataset(df, "sample_fraud_data")
        st.success(f"✅ Sample data loaded successfully! Database ID: {dataset_id}")
    
    # Display data analysis if data is loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Data overview
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Dataset Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            fraud_count = df['Class'].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            st.metric("Features", len(df.columns) - 1)
        
        # Data preview
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">👀 Data Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df.head(), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_class = px.pie(
                values=[len(df) - fraud_count, fraud_count],
                names=['Normal', 'Fraud'],
                title="Transaction Distribution",
                color_discrete_sequence=['#51cf66', '#ff6b6b']
            )
            st.plotly_chart(fig_class, use_container_width=True)
        
        with col2:
            fig_amount = px.histogram(
                df, x='Amount', color='Class',
                title="Transaction Amount Distribution",
                nbins=50
            )
            st.plotly_chart(fig_amount, use_container_width=True)
        
        # Correlation analysis
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🔗 Feature Correlation Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Select key features for correlation
        key_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 11)] + ['Class']
        corr_df = df[key_features].corr()
        
        fig_corr = px.imshow(
            corr_df,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def model_training():
    st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the Data Analysis section.")
        return
    
    fraud_detector = st.session_state.fraud_detector
    
    # Training configuration
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">⚙️ Training Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        use_smote = st.checkbox("Use SMOTE Balancing", value=True)
    
    with col3:
        scale_features = st.checkbox("Scale Features", value=True)
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Prepare data
                fraud_detector.prepare_data(
                    st.session_state.data,
                    test_size=test_size,
                    use_smote=use_smote,
                    scale_features=scale_features
                )
                
                # Train models
                fraud_detector.train_models()
                
                # Evaluate models
                results = fraud_detector.evaluate_models()
                
                st.session_state.models_trained = True
                st.session_state.evaluation_results = results
                
                # Store model performance in database
                db = st.session_state.database
                for model_name, metrics in results.items():
                    db.store_model_performance(
                        model_name, 
                        metrics, 
                        len(st.session_state.data),
                        st.session_state.data['Class'].sum()
                    )
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                display_training_results(fraud_detector, results)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
    
    # Display previous results if available
    if hasattr(st.session_state, 'evaluation_results'):
        display_training_results(fraud_detector, st.session_state.evaluation_results)

def display_training_results(fraud_detector, evaluation_results):
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Training Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics table
    metrics_df = pd.DataFrame(evaluation_results).T
    st.dataframe(metrics_df.round(4), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance comparison
        fig_metrics = px.bar(
            x=list(evaluation_results.keys()),
            y=[results['accuracy'] for results in evaluation_results.values()],
            title="Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy'}
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # ROC AUC comparison
        fig_roc = px.bar(
            x=list(evaluation_results.keys()),
            y=[results['roc_auc'] for results in evaluation_results.values()],
            title="ROC AUC Comparison",
            labels={'x': 'Model', 'y': 'ROC AUC'}
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Additional visualizations
    if hasattr(fraud_detector, 'models') and fraud_detector.models:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Advanced Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # ROC curves
            fraud_detector.plot_roc_curves()
            st.pyplot(plt.gcf(), clear_figure=True)
            
            # Confusion matrices
            fraud_detector.plot_confusion_matrices()
            st.pyplot(plt.gcf(), clear_figure=True)
            
        except Exception as e:
            st.info(f"Advanced visualizations not available: {str(e)}")

def fraud_prediction():
    st.markdown('<h2 class="section-header">🔍 Fraud Detection</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    fraud_detector = st.session_state.fraud_detector
    
    # Prediction method selection
    prediction_method = st.radio(
        "Select prediction method:",
        ["Single Transaction", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_method == "Single Transaction":
        manual_prediction(fraud_detector)
    else:
        batch_prediction(fraud_detector)

def manual_prediction(fraud_detector):
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">💳 Single Transaction Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    selected_model = st.selectbox(
        "Choose model for prediction:",
        list(fraud_detector.models.keys()),
        help="Select which trained model to use for prediction"
    )
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Transaction Time", value=0.0, help="Time in seconds from first transaction")
        amount = st.number_input("Transaction Amount", value=100.0, min_value=0.0, help="Transaction amount in currency units")
    
    with col2:
        st.write("**V Features (PCA Components)**")
        # Create V features input
        v_features = {}
        for i in range(1, 29):
            if i <= 14:
                v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')
    
    # Additional V features in expandable section
    with st.expander("Additional V Features (V15-V28)"):
        for i in range(15, 29):
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')
    
    if st.button("🔍 Analyze Transaction", type="primary"):
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
            
            # Display results
            is_fraud = prediction[0] == 1
            fraud_prob = probability[0][1] if len(probability[0]) > 1 else probability[0][0]
            
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-alert">
                    🚨 FRAUD DETECTED 🚨<br>
                    Confidence: {fraud_prob:.1%}<br>
                    Model: {selected_model}<br>
                    Prediction ID: {prediction_id}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    ✅ TRANSACTION APPROVED ✅<br>
                    Safety Score: {(1-fraud_prob):.1%}<br>
                    Model: {selected_model}<br>
                    Prediction ID: {prediction_id}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.1%}")
                st.metric("Normal Probability", f"{(1-fraud_prob):.1%}")
            
            with col2:
                st.metric("Risk Level", "HIGH" if fraud_prob > 0.5 else "LOW")
                st.metric("Model Used", selected_model)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def batch_prediction(fraud_detector):
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Batch Transaction Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload transactions for batch prediction",
        type=['csv'],
        help="CSV file with same format as training data (without Class column)"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df = preprocess_uploaded_data(batch_df)
            
            # Remove Class column if present
            if 'Class' in batch_df.columns:
                actual_labels = batch_df['Class']
                batch_df = batch_df.drop('Class', axis=1)
            else:
                actual_labels = None
            
            selected_model = st.selectbox(
                "Choose model for batch prediction:",
                list(fraud_detector.models.keys())
            )
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    predictions, probabilities = fraud_detector.predict(batch_df, selected_model)
                    
                    # Create results dataframe
                    results_df = batch_df.copy()
                    results_df['Prediction'] = predictions
                    results_df['Fraud_Probability'] = [prob[1] if len(prob) > 1 else prob[0] for prob in probabilities]
                    results_df['Risk_Level'] = ['HIGH' if p > 0.5 else 'LOW' for p in results_df['Fraud_Probability']]
                    
                    # Display summary
                    fraud_count = sum(predictions)
                    total_count = len(predictions)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Transactions", total_count)
                    
                    with col2:
                        st.metric("Fraud Detected", fraud_count)
                    
                    with col3:
                        st.metric("Fraud Rate", f"{(fraud_count/total_count)*100:.1f}%")
                    
                    # Results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    fig_pred = px.pie(
                        values=[total_count - fraud_count, fraud_count],
                        names=['Normal', 'Fraud'],
                        title="Batch Prediction Results"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Batch prediction failed: {str(e)}")

def model_comparison():
    st.markdown('<h2 class="section-header">📈 Performance Metrics</h2>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'evaluation_results'):
        st.warning("⚠️ No evaluation results available. Please train models first.")
        return
    
    results = st.session_state.evaluation_results
    
    # Performance summary
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🏆 Model Performance Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create comprehensive comparison
    metrics_df = pd.DataFrame(results).T
    
    # Best performing model for each metric
    best_models = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        best_models[metric] = metrics_df[metric].idxmax()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Best Accuracy",
            f"{metrics_df['accuracy'].max():.3f}",
            delta=f"{best_models['accuracy']}"
        )
    
    with col2:
        st.metric(
            "Best F1-Score",
            f"{metrics_df['f1'].max():.3f}",
            delta=f"{best_models['f1']}"
        )
    
    with col3:
        st.metric(
            "Best ROC-AUC",
            f"{metrics_df['roc_auc'].max():.3f}",
            delta=f"{best_models['roc_auc']}"
        )
    
    # Detailed metrics table
    st.dataframe(metrics_df.round(4), use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for model comparison
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig_radar = go.Figure()
        
        for model_name in results.keys():
            fig_radar.add_trace(go.Scatterpolar(
                r=[results[model_name][metric] for metric in metrics_to_plot],
                theta=metrics_to_plot,
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Metric comparison bar chart
        fig_bar = px.bar(
            x=list(results.keys()),
            y=[results[model]['f1'] for model in results.keys()],
            title="F1-Score Comparison",
            labels={'x': 'Model', 'y': 'F1-Score'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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

    # Footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-top: 3rem; text-align: center; color: white;">
        <h3 style="margin-bottom: 1rem;">🛡️ FraudGuard Pro</h3>
        <p style="margin-bottom: 0.5rem;">Enterprise-Grade Fraud Detection System</p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="color: #d0d0d0; font-size: 0.8rem; margin: 0;">© 2025 FraudGuard Pro. Protecting financial transactions with cutting-edge AI.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()