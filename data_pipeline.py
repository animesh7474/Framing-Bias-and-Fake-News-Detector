"""
data_pipeline.py — Multi-stage ETL pipeline for text data processing.
Domain: Big Data Analytics
"""

import pandas as pd
import os
from config import DATASET_PATH, REPORTS_DIR, TFIDF_MAX_FEATURES
from logger import get_logger

log = get_logger("data_pipeline")


class ETLPipeline:
    """
    Extract → Transform → Load pipeline for the framing bias dataset.
    Demonstrates Big Data ETL concepts using chunked batch processing.
    """

    CHUNK_SIZE = 500

    def __init__(self, filepath: str = DATASET_PATH):
        self.filepath = filepath
        self.df = None
        os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Stage 1: Extract ─────────────────────────────────────────────────────
    def extract(self):
        log.info(f"[EXTRACT] Reading dataset from {self.filepath}")
        chunks = []
        for i, chunk in enumerate(
            pd.read_csv(self.filepath, chunksize=self.CHUNK_SIZE)
        ):
            log.info(f"  → Chunk {i+1}: {len(chunk)} rows loaded")
            chunks.append(chunk)
        self.df = pd.concat(chunks, ignore_index=True)
        log.info(f"[EXTRACT] Total rows: {len(self.df)}")
        return self

    # ── Stage 2: Transform ───────────────────────────────────────────────────
    def transform(self):
        log.info("[TRANSFORM] Cleaning and enriching data...")
        self.df.dropna(subset=["text", "label"], inplace=True)
        self.df["text"] = self.df["text"].str.strip()
        self.df["text_length"] = self.df["text"].str.split().str.len()
        self.df["label"] = self.df["label"].str.strip()
        log.info(f"[TRANSFORM] Label distribution:\n{self.df['label'].value_counts().to_string()}")
        return self

    # ── Stage 3: Load ─────────────────────────────────────────────────────────
    def load(self, output_path: str = None):
        out = output_path or self.filepath
        self.df.to_csv(out, index=False)
        log.info(f"[LOAD] Cleaned dataset saved to {out} ({len(self.df)} rows)")
        return self.df

    def run(self):
        return self.extract().transform().load()


if __name__ == "__main__":
    pipe = ETLPipeline()
    df = pipe.run()
    print(f"\n✅ Pipeline complete. {len(df)} records processed.")
    print(df["label"].value_counts())
