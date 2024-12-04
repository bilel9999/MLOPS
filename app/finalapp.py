import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import joblib
import os

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
    
    # File uploaders
    st.header("📂 File Upload")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    # Folder for encoders and scalers
    encoder_folder = st.text_input("Enter path to encoder folder", 
                                   placeholder="e.g., /path/to/encoders")
    
    # MLflow model path
    model_uri = st.text_input("Enter MLflow Model URI", 
                               placeholder="models:/MicrosoftMalwareModel/Production")
    
    # Columns file path
    columns_file = st.file_uploader("Upload columns file (.pkl)", type=['pkl'])
    
    # Prediction button
    if st.button("🚀 Predict Malware"):
        if uploaded_file is not None and columns_file is not None and encoder_folder and model_uri:
            try:
                # 1. Load the predefined columns
                columns_to_use = joblib.load(columns_file)
                st.write("Columns to use:", columns_to_use)
                
                # 2. Read the CSV
                df = pd.read_csv(uploaded_file)
                
                # Ensure only selected columns are used
                df = df[columns_to_use]
                
                # 3. Apply Label Encoding
                # Iterate through all label encoder files
                for encoder_file in os.listdir(encoder_folder):
                    if encoder_file.endswith('_encoder.pkl'):
                        # Extract column name (assuming filename format: column_name_encoder.pkl)
                        col_name = encoder_file.replace('_encoder.pkl', '')
                        
                        # Check if column exists in dataframe
                        if col_name in df.columns:
                            # Load encoder
                            encoder_path = os.path.join(encoder_folder, encoder_file)
                            encoder = joblib.load(encoder_path)
                            
                            # Transform column
                            df[col_name] = encoder.transform(df[col_name].astype(str))
                
                # 4. Apply Standard Scaler
                scaler_path = os.path.join(encoder_folder, 'StandardScaler.pkl')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    
                    # Select numeric columns for scaling
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    df[numeric_cols] = scaler.transform(df[numeric_cols])
                
                # 5. Load MLflow Model and Predict
                model = mlflow.pyfunc.load_model(model_uri)
                
                # Make prediction
                predictions = model.predict(df)
                
                # Display results
                st.subheader("Prediction Results")
                results_df = pd.DataFrame({
                    'Predictions': predictions
                })
                st.dataframe(results_df)
                
                # Additional info
                st.write(f"Total predictions: {len(predictions)}")
                
            except Exception as e:
                st.error(f"Error in prediction process: {e}")
                st.exception(e)

# Run the app
if __name__ == "__main__":
    main()