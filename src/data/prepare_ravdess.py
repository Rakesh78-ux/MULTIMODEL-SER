import os
os.environ["LIBROSA_RES_TYPE"] = "kaiser_fast"
os.environ["LIBROSA_DISABLE_SOXR"] = "1"


import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/raw/RAVDESS"
OUTPUT_DIR = "data/processed/RAVDESS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map RAVDESS emotion codes
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path, sr=16000, n_mfcc=40):
    """Extract MFCC features from audio using only kaiser_fast"""
    try:
        # Force librosa to ignore soxr completely
        y, sr = librosa.load(file_path, sr=sr, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")
        return None



def prepare_dataset():
    print(f"🔍 Looking for dataset in {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ RAVDESS dataset not found at {DATA_DIR}")
        return

    data = []
    total_files = 0
    processed_files = 0

    actor_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    for actor_dir in tqdm(actor_dirs, desc="Actors"):
        print(f"➡️ Processing {actor_dir}")
        actor_path = os.path.join(DATA_DIR, actor_dir)

        for fname in os.listdir(actor_path):
            if fname.endswith(".wav"):
                total_files += 1
                parts = fname.split("-")
                if len(parts) < 3:
                    print(f"⚠️ Skipping malformed filename: {fname}")
                    continue
                emotion = parts[2]
                if emotion not in EMOTION_MAP:
                    continue

                label = EMOTION_MAP[emotion]
                file_path = os.path.join(actor_path, fname)

                features = extract_features(file_path)
                if features is not None:
                    processed_files += 1
                    data.append([file_path, label, features])

    if not data:
        print("❌ No features extracted. Please check dataset placement.")
        return

    df = pd.DataFrame(data, columns=["file", "label", "features"])
    output_path = os.path.join(OUTPUT_DIR, "ravdess_features.pkl")
    df.to_pickle(output_path)

    print(f"\n✅ Saved features to {output_path}")
    print(f"📊 Processed {processed_files}/{total_files} files successfully")
    print(df.head())


if __name__ == "__main__":
    prepare_dataset()
