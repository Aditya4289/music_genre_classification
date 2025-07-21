import librosa
import numpy as np
import joblib

# Load saved models
model = joblib.load("genre_classifier.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# File to classify
file_path = input("Enter path to the audio file: ")

try:
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    genre = label_encoder.inverse_transform(prediction)
    print(f"The predicted genre is: {genre[0]}")
except Exception as e:
    print("Error:", e)
