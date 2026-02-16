"""ONNX Runtime CPU sentiment classifier (Phase 6).

Runs quantized DistilRoBERTa-Finance model on ARM64 CPU.
Expected inference: ~50-100ms per headline on Pi 5.

Model files expected in model_dir:
    - model_quantized.onnx (INT8, preferred) or model.onnx (FP32 fallback)
    - tokenizer_config.json
    - vocab.txt or sentencepiece.bpe.model
    - special_tokens_map.json
"""

import os
from typing import Dict, List

from src.shared.inference.base import SentimentClassifier

DEFAULT_MODEL_DIR = os.environ.get(
    "SENTIMENT_MODEL_DIR",
    "/root/models/distilbert-finance-sentiment",
)


class OnnxCpuClassifier(SentimentClassifier):
    """DistilRoBERTa-Finance sentiment classifier using ONNX Runtime CPU.

    Loads a quantized ONNX model and its tokenizer at construction time.
    Designed for Pi 5 CPU inference (~50-100ms per headline).

    The model (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
    outputs 3 classes: negative (0), neutral (1), positive (2).
    """

    # Label mapping for the distilroberta-finetuned-financial-news model
    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Quantized model is exported as model_quantized.onnx by optimum
        model_path = os.path.join(model_dir, "model_quantized.onnx")
        if not os.path.exists(model_path):
            # Fall back to model.onnx for non-quantized exports
            model_path = os.path.join(model_dir, "model.onnx")
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    def classify(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify texts using ONNX Runtime on CPU.

        Tokenizes in batch, runs single inference pass, applies softmax.

        Args:
            texts: List of headline/summary strings.

        Returns:
            List of sentiment probability dicts.
        """
        import numpy as np

        if not texts:
            return []

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        # Run inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self._session.run(None, ort_inputs)[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        results = []
        for prob_row in probs:
            result = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            for idx, label in self.LABEL_MAP.items():
                if idx < len(prob_row):
                    result[label] = float(round(prob_row[idx], 6))
            results.append(result)

        return results
