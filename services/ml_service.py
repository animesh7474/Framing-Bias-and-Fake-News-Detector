"""
ml_service.py — Service layer for ML model inference and lifecycle management.
Domain: Big Data Analytics + AI Cybersecurity
"""

import os
import joblib
import threading
from config import MODEL_PATH, FRAME_LABELS
from logger import get_logger

log = get_logger("ml_service")

class ModelManager:
    """
    Singleton class to manage the ML model in memory.
    Ensures the model is loaded only once and shared across requests.
    """
    _instance = None
    _model = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._version = "1.0"
        return cls._instance

    def _load_model(self):
        """Internal method to load the model from disk."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if os.path.exists(MODEL_PATH):
                        log.info(f"Loading ML model from {MODEL_PATH}...")
                        try:
                            self._model = joblib.load(MODEL_PATH)
                            log.info("Model loaded successfully into memory.")
                        except Exception as e:
                            log.error(f"Failed to load model: {e}")
                    else:
                        log.warning(f"Model file not found at {MODEL_PATH}")
        return self._model

    def get_model(self):
        """Returns the cached model instance, loading it if necessary."""
        return self._load_model()

    def reload(self):
        """Force a reload of the model from disk (useful after retraining)."""
        with self._lock:
            self._model = None
            self._version = self.get_version()
            return self._load_model()

    def get_version(self):
        """Determines version based on count of files in models directory."""
        from config import MODELS_DIR
        if not os.path.exists(MODELS_DIR):
            return "1.0"
        pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        return f"{len(pkl_files)+1}.0" if pkl_files else "1.0"

    @property
    def current_version(self):
        if not hasattr(self, '_version') or self._version == "1.0":
            self._version = self.get_history_version()
        return self._version

    def get_history_version(self):
        from config import MODELS_DIR
        if not os.path.exists(MODELS_DIR): return "1.0"
        pkls = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        return f"{len(pkls) + 1}.0"

def predict(text: str):
    """
    Performs ML inference using the cached model.
    Returns: (predicted_frame, confidence, all_confidences)
    """
    manager = ModelManager()
    model = manager.get_model()

    if not model:
        log.warning("No model available for prediction. Using fallback.")
        return "neutral", 0.5, {"neutral": 0.5}

    try:
        # Logistic Regression pipeline typically supports predict_proba
        probs = model.predict_proba([text])[0]
        # Get frame names from the model classes or config
        classes = model.classes_
        
        # Build confidence map
        conf_map = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        
        # Get top prediction
        top_idx = probs.argmax()
        pred = str(classes[top_idx])
        conf = float(probs[top_idx])

        return pred, conf, conf_map, manager.get_history_version()
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return "neutral", 0.5, {"neutral": 0.5}, manager.get_history_version()

def get_explainable_keywords(text: str, predicted_frame: str):
    """
    Simulates LIME-style word importance based on domain keywords.
    """
    words = text.split()
    frame_keywords = {
        "Economic":    ["economy","inflation","tax","budget","market","trade","jobs","recession","financial","investment","growth","cost"],
        "Political":   ["government","election","policy","senate","vote","parliament","law","campaign","legislation","president","minister"],
        "Social":      ["community","education","health","equality","rights","welfare","family","culture","discrimination","protest","society"],
        "Security":    ["war","military","conflict","crime","border","attack","defense","threat","terrorism","violence","forces","troops","nuclear"],
        "Environment": ["climate","pollution","energy","wildlife","conservation","warming","carbon","emissions","nature","forest","drought","flood"],
    }
    
    fkws = frame_keywords.get(predicted_frame, [])
    lime_words = []
    import random
    for w in words:
        c = w.lower().strip(".,!?\"'")
        score = 0.4 + round(random.random() * 0.4, 3) if any(k in c for k in fkws) else round(random.random() * 0.2 - 0.1, 3)
        lime_words.append({"word": w, "score": score})
    return lime_words

# Helper to pre-load model on startup
def bootstrap():
    ModelManager().get_model()
