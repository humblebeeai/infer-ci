# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
from .ci_val import DetectionCIValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionCIValidator"
