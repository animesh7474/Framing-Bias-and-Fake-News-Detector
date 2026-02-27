"""
stream_simulator.py — Real-time streaming simulation of text classification.
Domain: Big Data Analytics
"""

import time
import random
import joblib
import pandas as pd
from datetime import datetime
from config import MODEL_PATH, DATASET_PATH, STREAM_DELAY_SECONDS
from logger import get_logger

log = get_logger("stream")


def stream_predict(n_samples: int = 20):
    """
    Simulates a real-time data stream where texts arrive one-by-one
    and are classified as they come in. Demonstrates streaming analytics.
    """
    log.info("Loading model for stream processing...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH).sample(frac=1, random_state=42).reset_index(drop=True)

    correct = 0
    total = 0
    frame_counts = {}

    print("\n" + "═" * 65)
    print("  📡 REAL-TIME FRAMING BIAS STREAM PROCESSOR")
    print("═" * 65)
    print(f"  {'TIME':<10} {'PREDICTION':<15} {'CONFIDENCE':<12} TEXT")
    print("─" * 65)

    for i, row in df.head(n_samples).iterrows():
        text  = row["text"]
        true_label = row["label"]

        proba = model.predict_proba([text])[0]
        pred_idx = proba.argmax()
        pred  = model.classes_[pred_idx]
        conf  = proba[pred_idx]

        is_correct = (pred == true_label)
        if is_correct:
            correct += 1
        total += 1
        frame_counts[pred] = frame_counts.get(pred, 0) + 1

        ts = datetime.now().strftime("%H:%M:%S")
        tick = "✓" if is_correct else "✗"
        snippet = text[:38] + "..." if len(text) > 38 else text
        print(f"  {ts:<10} {pred:<15} {conf*100:>5.1f}%  {tick}  {snippet}")

        time.sleep(STREAM_DELAY_SECONDS)

    rolling_acc = correct / total * 100
    print("─" * 65)
    print(f"\n  📊 Stream Summary:")
    print(f"     Records processed : {total}")
    print(f"     Rolling accuracy  : {rolling_acc:.1f}%")
    print(f"     Frame distribution: {frame_counts}")
    print("═" * 65 + "\n")
    log.info(f"Stream complete. {total} records | Accuracy: {rolling_acc:.1f}%")


if __name__ == "__main__":
    stream_predict(n_samples=15)
