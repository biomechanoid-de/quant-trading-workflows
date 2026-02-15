"""Abstract base class for sentiment classifiers.

Inference-Abstraction-Layer: Build first, swap backend later.
Phase 6: OnnxCpuClassifier (ONNX Runtime on CPU)
Phase 7: HailoNpuClassifier (Hailo-10H NPU via HailoRT)
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class SentimentClassifier(ABC):
    """Abstract classifier that produces sentiment scores from text.

    Subclasses implement classify() for specific backends:
    - OnnxCpuClassifier: ONNX Runtime on CPU (Phase 6)
    - HailoNpuClassifier: Hailo-10H NPU via HailoRT (future)
    """

    @abstractmethod
    def classify(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify a batch of texts into sentiment probabilities.

        Args:
            texts: List of text strings (headlines/summaries).

        Returns:
            List of dicts, each with keys:
                - positive: float (0.0-1.0)
                - neutral: float (0.0-1.0)
                - negative: float (0.0-1.0)
            One dict per input text. Probabilities sum to ~1.0.
        """
        ...
