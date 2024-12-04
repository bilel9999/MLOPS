import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import seaborn as sns
import matplotlib.pyplot as plt

# Custom CSS for cybersecurity theme
def set_custom_theme():
    st.markdown("""
    <style>
    /* Cybersecurity-inspired theme */
    body {
        background-color: #0a0a1a;
        color: #00ff00;
    }
    .stApp {
        background-color: #0a0a1a;
    }
    .stButton>button {
        background-color: #00ff00;
        color: #0a0a1a;
        border: 2px solid #00ff00;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0a0a1a;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a2a;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    .stSelectbox>div>div>select {
        background-color: #1a1a2a;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    .stDataFrame {
        background-color: #1a1a2a;
        color: #00ff00;
    }
    .stHeader {
        color: #00ff00;
    }
    .stSubheader {
        color: #00ff00;
    }
    /* Hacker-style progress bar */
    .stProgress>div>div>div {
        background-color: #00ff00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Malware Detection App
def main():
    # Set custom theme
    set_custom_theme()
    
    # App title with hacker-style animation
    st.markdown("""
    <h1 style='text-align: center; color: #00ff00; 
    text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00;
    animation: glitch 1s linear infinite;'>
    🔒 MALWARE DETECTION SYSTEM 🛡️
    </h1>
    <style>
    @keyframes glitch {
        2%, 64% { transform: translate(2px, 0) skew(0deg); }
        4%, 60% { transform: translate(-2px, 0) skew(0deg); }
        62% { transform: translate(0, 0) skew(5deg); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🖥️ Malware Detection Workflow")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    # Workflow steps
    workflow_steps = [
        "1. Upload Dataset",
        "2. Data Exploration",
        "3. Preprocessing",
        "4. Model Prediction"
    ]
    current_step = st.sidebar.radio("Workflow Steps", workflow_steps)

    # Step 1: Upload Dataset
    if current_step == "1. Upload Dataset":
        st.header("📂 Dataset Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Basic dataset info
                st.subheader("Dataset Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Shape:** {df.shape}")
                with col2:
                    st.write(f"**Columns:** {list(df.columns)}")
                
            except Exception as e:
                st.error(f"Error reading the file: {e}")

    # Step 2: Data Exploration
    elif current_step == "2. Data Exploration":
        if st.session_state.data is not None:
            st.header("🔍 Data Exploration")
            df = st.session_state.data
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing Values
                st.subheader("Missing Values")
                missing_data = df.isnull().sum()
                st.bar_chart(missing_data)
            
            with col2:
                # Correlation Heatmap
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
                            cmap='coolwarm', ax=ax, annot=True, fmt='.2f')
                st.pyplot(fig)
            
            # Column Selection for Preprocessing
            st.subheader("Select Columns for Preprocessing")
            columns_to_preprocess = st.multiselect(
                "Choose columns to keep or encode", 
                list(df.columns)
            )
            
            if st.button("Proceed to Preprocessing"):
                # Create preprocessed dataset
                preprocessed_df = df[columns_to_preprocess].copy()
                st.session_state.preprocessed_data = preprocessed_df
                st.success("Preprocessing data prepared!")

    # Step 3: Preprocessing
    elif current_step == "3. Preprocessing":
        if st.session_state.preprocessed_data is not None:
            st.header("🛠️ Data Preprocessing")
            df = st.session_state.preprocessed_data
            
            # Encoding Options
            st.subheader("Encoding Options")
            encoding_method = st.selectbox(
                "Select Encoding Method", 
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            # Columns to encode
            columns_to_encode = st.multiselect(
                "Select columns to encode", 
                list(df.select_dtypes(include=['object']).columns)
            )
            
            # Preprocessing button
            if st.button("Apply Preprocessing"):
                # Perform encoding
                preprocessed_df = df.copy()
                
                if encoding_method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in columns_to_encode:
                        preprocessed_df[col] = le.fit_transform(df[col].astype(str))
                
                elif encoding_method == "One-Hot Encoding":
                    preprocessed_df = pd.get_dummies(
                        preprocessed_df, 
                        columns=columns_to_encode
                    )
                
                # Update session state
                st.session_state.preprocessed_data = preprocessed_df
                
                # Display preprocessed data
                st.subheader("Preprocessed Data")
                st.dataframe(preprocessed_df.head())
                
                st.success("Preprocessing Complete!")

    # Step 4: Model Prediction
    elif current_step == "4. Model Prediction":
        if st.session_state.preprocessed_data is not None:
            st.header("🤖 Model Prediction")
            
            # MLflow Model Loading
            st.subheader("Load MLflow Model")
            model_uri = st.text_input(
                "Enter MLflow Model URI", 
                "models:/MicrosoftMalwareModel/Production"
            )
            
            # Target Column Selection
            target_column = st.selectbox(
                "Select Target Column", 
                list(st.session_state.preprocessed_data.columns)
            )
            
            # Prepare features and target
            X = st.session_state.preprocessed_data.drop(columns=[target_column])
            y = st.session_state.preprocessed_data[target_column]
            
            # Prediction Button
            if st.button("🚀 Run Prediction"):
                try:
                    # Load MLflow Model
                    model = mlflow.pyfunc.load_model(model_uri)
                    
                    # Make Predictions
                    predictions = model.predict(X)
                    
                    # Display Results
                    st.subheader("Prediction Results")
                    results_df = pd.DataFrame({
                        'Actual': y,
                        'Predicted': predictions
                    })
                    st.dataframe(results_df)
                    
                    # Classification Metrics
                    from sklearn.metrics import (
                        accuracy_score, precision_score, 
                        recall_score, f1_score, confusion_matrix
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy_score(y, predictions):.2%}")
                    col2.metric("Precision", f"{precision_score(y, predictions, average='weighted'):.2%}")
                    col3.metric("Recall", f"{recall_score(y, predictions, average='weighted'):.2%}")
                    col4.metric("F1 Score", f"{f1_score(y, predictions, average='weighted'):.2%}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y, predictions)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

# Run the app
if __name__ == "__main__":
    main()