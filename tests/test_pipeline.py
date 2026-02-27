"""
tests/test_pipeline.py — Unit tests for the framing bias detection pipeline.
Domain: Software Development & Operations
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import joblib

# ─── Test Data ────────────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Economic":    "The economy is suffering due to rising inflation and high tax rates.",
    "Political":   "The government passed a new election reform bill in parliament.",
    "Social":      "Community leaders are calling for equality and social justice.",
    "Security":    "Military forces responded to the border conflict with increased patrols.",
    "Environment": "Climate change is causing severe droughts and forest fires worldwide.",
}

# ─── Tests ────────────────────────────────────────────────────────────────────
def test_dataset_exists():
    assert os.path.exists("dataset.csv"), "dataset.csv not found"

def test_dataset_columns():
    df = pd.read_csv("dataset.csv")
    assert "text" in df.columns
    assert "label" in df.columns

def test_dataset_size():
    df = pd.read_csv("dataset.csv")
    assert len(df) >= 100, "Dataset too small"

def test_model_exists():
    assert os.path.exists("framing_bias_model.pkl"), "Model file not found"

def test_model_predict():
    model = joblib.load("framing_bias_model.pkl")
    texts = list(SAMPLE_TEXTS.values())
    preds = model.predict(texts)
    assert len(preds) == len(texts)

def test_model_predict_proba():
    model = joblib.load("framing_bias_model.pkl")
    proba = model.predict_proba(["The stock market crashed today."])
    assert proba.shape[1] == 5  # 5 frames

def test_config_imports():
    from config import MODEL_PATH, DATASET_PATH, FRAME_LABELS
    assert len(FRAME_LABELS) == 5

def test_threat_analyzer():
    from threat_analyzer import ThreatAnalyzer
    ta = ThreatAnalyzer()
    result = ta.analyze("The government is destroying our economy with terrible policies!")
    assert "threat_level" in result
    assert "manipulation_score" in result
    assert result["manipulation_score"] >= 0
