# 🎵 Music Genre Classification 🎧

This project is a machine learning-based music genre classifier that takes a `.wav` audio file as input and predicts its genre (e.g., rock, jazz, etc.) using MFCC features extracted from the audio and a trained classification model.

---

## 📌 Features

- Upload a `.wav` music file (30 sec recommended)
- Extract audio features using `librosa`
- Predict genre using a trained ML model
- Web interface built with **Streamlit**
- Deployable to **Streamlit Community Cloud**

---

## 💻 Tech Stack

- Python
- Streamlit
- Librosa (Audio Feature Extraction)
- Scikit-learn (Model Training)
- Joblib (Model Serialization)
- NumPy & Pandas

---

## 📁 Project Structure

music_genre_classification/
│
├── app.py # Streamlit web app
├── extract_features.py # Feature extraction script
├── model_train.py # Model training script
├── genre_classifier.pkl # Trained classification model
├── scaler.pkl # Scaler for preprocessing
├── label_encoder.pkl # Label encoder
├── music_features.csv # Extracted features from audio files
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🚀 Local Setup Instructions

### 1️⃣ Clone the repository

git clone https://github.com/aditya4289/music-genre-classification.git
cd music-genre-classification

---

### 2️⃣ Install dependencies

Make sure you have Python 3.8+ installed.

pip install -r requirements.txt

---

### 3️⃣ Extract features from .wav dataset

python extract_features.py

---

### 4️⃣ Train the model

python model_train.py

---

### 5️⃣ Run the Streamlit web app

streamlit run app.py

Then open http://localhost:8501 in your browser.

--- 

🧠 Notes
Recommended .wav file length: 30 seconds

Avoid corrupted or empty audio files

For best results, train the model on a balanced dataset with multiple genres
