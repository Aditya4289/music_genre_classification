import os
import librosa
import numpy as np
import pandas as pd

# Set the path to your dataset
DATASET_PATH = "genres"
GENRES = os.listdir(DATASET_PATH)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

features_list = []

for genre in GENRES:
    genre_path = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(genre_path, filename)
            try:
                data = extract_features(file_path)
                features_list.append([*data, genre])
                print(f"Processed {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Create DataFrame
df = pd.DataFrame(features_list)
columns = [f"mfcc_{i+1}" for i in range(13)] + ["label"]
df.columns = columns
df.to_csv("music_features.csv", index=False)
print("Saved features to music_features.csv")
