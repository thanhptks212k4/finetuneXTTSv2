"""
xtts_finetune - XTTS v2 Vietnamese Fine-Tuning Pipeline
"""

from .config import TrainingConfig
from .utils import get_logger, set_seed, free_memory, normalize_vietnamese_text
from .dataset import load_manifest, validate_and_filter, split_into_patches
from .model_loader import (
    download_base_model,
    load_xtts_model,
    configure_trainable_params,
    save_checkpoint,
    load_checkpoint,
)
from .trainer import XTTSTrainer
from .inference import run_inference

__version__ = "1.0.0"
__all__ = [
    "TrainingConfig",
    "XTTSTrainer",
    "run_inference",
    "get_logger",
    "set_seed",
    "free_memory",
    "normalize_vietnamese_text",
    "load_manifest",
    "validate_and_filter",
    "split_into_patches",
    "download_base_model",
    "load_xtts_model",
    "configure_trainable_params",
    "save_checkpoint",
    "load_checkpoint",
]
