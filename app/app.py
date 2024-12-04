import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow.pyfunc
import os
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
    .stProgress>div>div>div {
        background-color: #00ff00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Malware Detection App
def main():
    # Set custom theme
    set_custom_theme()

    # App title
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>🔒 MALWARE DETECTION SYSTEM 🛡️</h1>", unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🖥️ Malware Detection Workflow")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None

    # Workflow steps
    workflow_steps = ["1. Upload Dataset", "2. Preprocess and Predict"]
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

    # Step 2: Preprocess and Predict
    elif current_step == "2. Preprocess and Predict":
        if st.session_state.data is not None:
            st.header("🛠️ Preprocessing and Prediction")

            # Load final column structure
            try:
                clean_cols = joblib.load("clean_df_cols.pkl")
                st.success("Final column structure loaded.")
            except Exception as e:
                st.error(f"Error loading clean_df_cols.pkl: {e}")
                return

            # Keep only relevant columns
            df = st.session_state.data
            if not all(col in df.columns for col in clean_cols):
                st.error("Uploaded dataset is missing required columns.")
                return
            df = df[clean_cols]

            # Apply label encoding
            encoders_dir = "encoders"  # Specify the directory containing .pkl encoders
            for col in df.columns:
                encoder_path = os.path.join(encoders_dir, f"{col}_encoder.pkl")
                if os.path.exists(encoder_path):
                    try:
                        encoder = joblib.load(encoder_path)
                        df[col] = encoder.transform(df[col].astype(str))
                        st.success(f"Applied encoding for column: {col}")
                    except Exception as e:
                        st.error(f"Error applying encoder for column {col}: {e}")
                        return

            # Apply standard scaling
            try:
                scaler = joblib.load("StandardScaler.pkl")
                scaled_data = scaler.transform(df)
                st.success("Data successfully scaled.")
            except Exception as e:
                st.error(f"Error loading StandardScaler.pkl: {e}")
                return

            # Load MLflow model
            model_uri = st.text_input(
                "Enter MLflow Model URI",
                "mlflow-artifacts:/1c1380b39e9248c6ad4366557dd0df2e/6abc95022e654bf0b12ad2dbd464083f/artifacts/ML_models"
            )
            if st.button("🚀 Run Prediction"):
                try:
                    model = mlflow.pyfunc.load_model(model_uri)
                    predictions = model.predict(scaled_data)

                    # Display results
                    st.subheader("Prediction Results")
                    results_df = pd.DataFrame(predictions, columns=['Predicted Label'])
                    st.dataframe(results_df)

                except Exception as e:
                    st.error(f"Error in prediction: {e}")

# Run the app
if __name__ == "__main__":
    main()
