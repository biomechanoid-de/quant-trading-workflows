"""Tests for M2: Sentiment classifier abstraction + ONNX CPU.

All ONNX sessions and tokenizers are mocked — no model files needed.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.shared.inference.base import SentimentClassifier
from src.shared.inference.onnx_cpu import OnnxCpuClassifier


# ============================================================
# SentimentClassifier ABC
# ============================================================

class TestSentimentClassifierABC:
    """Verify abstract base class cannot be instantiated."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SentimentClassifier()

    def test_subclass_must_implement_classify(self):
        class IncompleteClassifier(SentimentClassifier):
            pass

        with pytest.raises(TypeError):
            IncompleteClassifier()

    def test_concrete_subclass_works(self):
        class MockClassifier(SentimentClassifier):
            def classify(self, texts):
                return [{"positive": 0.8, "neutral": 0.1, "negative": 0.1}] * len(texts)

        clf = MockClassifier()
        result = clf.classify(["test"])
        assert len(result) == 1
        assert result[0]["positive"] == 0.8


# ============================================================
# OnnxCpuClassifier
# ============================================================

def _make_classifier(logits=None):
    """Helper to create a classifier with mocked dependencies (no __init__)."""
    clf = object.__new__(OnnxCpuClassifier)

    mock_tokenizer = MagicMock()

    def tokenize_fn(texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": np.ones((n, 10), dtype=np.int64),
            "attention_mask": np.ones((n, 10), dtype=np.int64),
        }
    mock_tokenizer.side_effect = tokenize_fn

    mock_session = MagicMock()
    if logits is not None:
        mock_session.run.return_value = [logits]

    clf._tokenizer = mock_tokenizer
    clf._session = mock_session

    return clf, mock_session, mock_tokenizer


class TestOnnxCpuClassifier:
    """Tests for ONNX Runtime CPU classifier with mocked session."""

    def test_classify_empty_input(self):
        clf, _, _ = _make_classifier()
        result = clf.classify([])
        assert result == []

    def test_classify_single_positive(self):
        # Logits: [negative, neutral, positive] = [-2.0, 0.0, 3.0]
        logits = np.array([[-2.0, 0.0, 3.0]])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["Apple beats earnings"])

        assert len(result) == 1
        assert result[0]["positive"] > result[0]["neutral"]
        assert result[0]["positive"] > result[0]["negative"]

    def test_classify_single_negative(self):
        logits = np.array([[3.0, 0.0, -2.0]])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["Massive layoffs announced"])

        assert len(result) == 1
        assert result[0]["negative"] > result[0]["positive"]

    def test_classify_batch_of_three(self):
        logits = np.array([
            [-2.0, 0.0, 3.0],   # positive
            [0.0, 3.0, 0.0],    # neutral
            [3.0, 0.0, -2.0],   # negative
        ])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["good", "meh", "bad"])

        assert len(result) == 3
        assert result[0]["positive"] > 0.5
        assert result[1]["neutral"] > 0.5
        assert result[2]["negative"] > 0.5

    def test_softmax_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["test"])

        total = result[0]["positive"] + result[0]["neutral"] + result[0]["negative"]
        assert abs(total - 1.0) < 1e-5

    def test_label_mapping(self):
        assert OnnxCpuClassifier.LABEL_MAP == {0: "negative", 1: "neutral", 2: "positive"}

    def test_classify_all_keys_present(self):
        logits = np.array([[0.0, 0.0, 0.0]])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["test"])

        assert "positive" in result[0]
        assert "neutral" in result[0]
        assert "negative" in result[0]

    def test_classify_equal_logits_gives_uniform(self):
        logits = np.array([[0.0, 0.0, 0.0]])
        clf, _, _ = _make_classifier(logits=logits)

        result = clf.classify(["test"])

        # All equal logits -> uniform distribution -> each ~0.333
        assert abs(result[0]["positive"] - 1.0 / 3.0) < 0.01
        assert abs(result[0]["neutral"] - 1.0 / 3.0) < 0.01
        assert abs(result[0]["negative"] - 1.0 / 3.0) < 0.01
