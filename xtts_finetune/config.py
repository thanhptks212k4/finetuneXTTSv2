"""
config.py - Central configuration for XTTS v2 Vietnamese fine-tuning pipeline.
All hyperparameters, paths, and flags are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class TrainingConfig:
    # ─── Paths ────────────────────────────────────────────────────────────────
    # HuggingFace repo for base model snapshot
    hf_repo_id: str = "anhnh2002/vnTTS"

    # Local directory where the HF snapshot is cached / downloaded
    base_model_dir: str = "./base_model"

    # XTTS checkpoint files (relative to base_model_dir)
    model_checkpoint: str = "model.pth"
    config_file: str = "config.json"
    vocab_file: str = "vocab.json"
    dvae_checkpoint: str = "dvae.pth"
    mel_stats_file: str = "mel_stats.pth"

    # Dataset manifest (JSONL)
    train_manifest: str = "./data/train.jsonl"
    val_manifest: str = "./data/val.jsonl"

    # Audio path remapping: fix broken absolute paths baked into manifest files.
    # Example: the manifest was created on a different machine / Kaggle dataset path.
    # Format: {"/old/prefix": "/new/prefix"}
    # Set to None to disable.
    audio_root_remap: Optional[dict] = None

    # Output directories
    output_dir: str = "./output"
    checkpoint_dir: str = "./output/checkpoints"
    sample_dir: str = "./output/samples"
    log_dir: str = "./output/logs"

    # ─── Audio ────────────────────────────────────────────────────────────────
    sample_rate: int = 22050          # XTTS internal sample rate
    max_audio_length: float = 20.0    # seconds — filter out longer clips
    min_audio_length: float = 1.0     # seconds — filter out shorter clips

    # ─── Dataset / Patching ───────────────────────────────────────────────────
    patch_size: int = 5000            # samples per patch
    resume_patch: int = 0             # set > 0 to resume from patch N
    num_workers: int = 2              # DataLoader workers

    # ─── Speaker ──────────────────────────────────────────────────────────────
    # "single" → use one reference audio for all samples
    # "multi"  → each sample has its own speaker embedding
    speaker_mode: str = "single"      # "single" | "multi"
    reference_audio: str = "./data/reference.wav"  # used in single-speaker mode

    # ─── Training ─────────────────────────────────────────────────────────────
    epochs_per_patch: int = 1         # epochs to train on each patch
    batch_size: int = 4               # per-GPU batch size
    grad_accum_steps: int = 4         # effective batch = batch_size * grad_accum
    learning_rate: float = 2e-5
    lr_min: float = 1e-6
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Mixed precision
    use_fp16: bool = True

    # Gradient checkpointing (saves VRAM at cost of speed)
    gradient_checkpointing: bool = True

    # Freeze encoder layers (True = only train decoder + speaker components)
    freeze_encoder: bool = True

    # ─── Loss weights ─────────────────────────────────────────────────────────
    mel_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1
    speaker_loss_weight: float = 0.1

    # ─── Evaluation ───────────────────────────────────────────────────────────
    eval_every_n_steps: int = 500     # run eval every N optimizer steps
    save_every_n_steps: int = 500
    val_texts: List[str] = field(default_factory=lambda: [
        "Xin chào, đây là hệ thống chuyển văn bản thành giọng nói.",
        "Hôm nay trời đẹp, tôi rất vui được gặp bạn.",
        "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.",
    ])

    # ─── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"              # "cuda" | "cpu"
    zip_checkpoints: bool = False     # zip each checkpoint (useful on Kaggle)

    # LoRA (bonus)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj"
    ])

    def __post_init__(self):
        """Create output directories on init."""
        for d in [self.output_dir, self.checkpoint_dir, self.sample_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    @property
    def model_path(self) -> str:
        return os.path.join(self.base_model_dir, self.model_checkpoint)

    @property
    def config_path(self) -> str:
        return os.path.join(self.base_model_dir, self.config_file)

    @property
    def vocab_path(self) -> str:
        return os.path.join(self.base_model_dir, self.vocab_file)

    @property
    def dvae_path(self) -> str:
        return os.path.join(self.base_model_dir, self.dvae_checkpoint)

    @property
    def mel_stats_path(self) -> str:
        return os.path.join(self.base_model_dir, self.mel_stats_file)
