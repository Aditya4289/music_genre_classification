import streamlit as st
import librosa
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")

# Load model, scaler, and label encoder
try:
    model = joblib.load("genre_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Error loading model/scaler/encoder: {e}")
    st.stop()

# Feature extraction function
def extract_features(file):
    try:
        y, sr = librosa.load(file, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# App UI
st.title("🎧 Music Genre Classifier")
st.markdown("Upload a `.wav` file and get the predicted **music genre** using a trained ML model.")

uploaded_file = st.file_uploader("🎼 Upload your .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("🔍 Extracting features and predicting..."):
        features = extract_features(uploaded_file)
        if features is not None:
            try:
                scaled = scaler.transform([features])
                prediction = model.predict(scaled)
                genre = encoder.inverse_transform(prediction)[0]
                st.success(f"✅ **Predicted Genre:** 🎵 {genre.upper()}")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
