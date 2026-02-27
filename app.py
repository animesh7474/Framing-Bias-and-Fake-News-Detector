"""
app.py — Flask REST API exposing the full ML+LLM analysis pipeline.
Domain: Computer Networks
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pandas as pd, json, os
from datetime import datetime
from dotenv import load_dotenv

from config import MODEL_PATH, DATASET_PATH, RESULTS_CSV, API_HOST, API_PORT, DEBUG_MODE
from logger import get_logger
from threat_analyzer import ThreatAnalyzer
from analytics_dashboard import run_analytics
from pipeline import run_full_analysis

load_dotenv()
log = get_logger("api")
app = Flask(__name__)
CORS(app)

model = None
threat_analyzer = ThreatAnalyzer()


def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            log.info("Model loaded.")
        else:
            log.warning("Model not found. Run framing_bias_detector.py first.")
    return model


@app.before_request
def log_request():
    log.info(f"→ {request.method} {request.path} from {request.remote_addr}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    m = get_model()
    groq_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    return jsonify({
        "status": "ok",
        "model_loaded": m is not None,
        "llm_available": groq_key,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


# ── Full Pipeline: ML + News + LLM ───────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    POST /api/analyze  {text: "..."}
    Full 4-stage pipeline: ML Model → News Fetch → LLM → Combined response
    """
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Field 'text' is required"}), 400

    try:
        result = run_full_analysis(data["text"].strip())
        return jsonify(result)
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Fast ML-only predict ──────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Field 'text' is required"}), 400

    text = data["text"].strip()
    m = get_model()
    if m is None:
        return jsonify({"error": "Model not loaded"}), 503

    proba  = m.predict_proba([text])[0]
    labels = list(m.classes_)
    pred   = labels[proba.argmax()]
    conf   = float(proba.max())
    threat = threat_analyzer.analyze(text)

    return jsonify({
        "text": text,
        "predicted_frame": pred,
        "confidence": round(conf, 4),
        "all_confidences": {labels[i]: round(float(proba[i]), 4) for i in range(len(labels))},
        "threat_analysis": threat,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


# ── Models leaderboard ────────────────────────────────────────────────────────
@app.route("/api/models", methods=["GET"])
def models():
    if not os.path.exists(RESULTS_CSV):
        return jsonify({"error": "Run framing_bias_detector.py first."}), 404
    df = pd.read_csv(RESULTS_CSV)
    return jsonify({"models": df.sort_values("Accuracy", ascending=False).to_dict(orient="records")})


# ── Dataset stats ─────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found."}), 404
    df = pd.read_csv(DATASET_PATH)
    return jsonify({
        "total_records": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "avg_text_length_words": round(float(df["text"].str.split().str.len().mean()), 2),
        "frames": list(df["label"].unique()),
    })


# ── Security log ──────────────────────────────────────────────────────────────
@app.route("/api/security-log", methods=["GET"])
def security_log():
    from config import SECURITY_LOG
    if not os.path.exists(SECURITY_LOG):
        return jsonify({"logs": []})
    with open(SECURITY_LOG, "r", encoding="utf-8") as f:
        return jsonify({"logs": json.load(f)[-50:]})


# ── Analytics charts ──────────────────────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
def analytics():
    try:
        return jsonify({"status": "ok", "charts": run_analytics()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):  return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    log.info(f"Starting Flask API on {API_HOST}:{API_PORT}")
    get_model()
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
