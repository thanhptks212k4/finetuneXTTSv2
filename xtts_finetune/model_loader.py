"""
model_loader.py - Download, load, and configure the XTTS v2 model
                  from the HuggingFace snapshot (anhnh2002/vnTTS).

Uses Coqui TTS internal APIs — NOT HuggingFace Transformers.
"""

import os
import gc
import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from .config import TrainingConfig
from .utils import get_logger, free_memory


# ─── HuggingFace Snapshot Download ────────────────────────────────────────────

def _ensure_numpy_compat():
    """
    Kaggle Python 3.12 ships numpy 2.x but scipy/sklearn/transformers
    were compiled against numpy 1.x → ImportError on '_center' etc.
    Downgrade numpy to 1.26.4 in-process if needed.
    This must run BEFORE any TTS / transformers import.
    
    NOTE: For automated notebooks, we downgrade but continue execution.
    The downgrade takes effect for subprocess imports (TTS library).
    """
    import subprocess, sys, importlib
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            print(f"⚠️  numpy {np.__version__} detected — downgrading to 1.26.4 for Kaggle compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q",
                 "numpy==1.26.4", "--force-reinstall"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Invalidate caches so new imports use 1.26.4
            importlib.invalidate_caches()
            print("✅ numpy downgraded to 1.26.4 (will take effect for TTS imports)")
            # DO NOT raise error - continue execution
            # The downgrade is effective for subprocess imports
    except Exception as e:
        # If downgrade fails, log but continue
        print(f"⚠️  numpy downgrade failed: {e}")
        print("Continuing anyway - may encounter import errors...")
        pass


def download_base_model(config: TrainingConfig, logger: logging.Logger) -> str:
    """
    Download the XTTS v2 checkpoint from HuggingFace Hub if not already present.
    Returns the local directory path.
    """
    from huggingface_hub import snapshot_download

    required_files = [
        config.model_checkpoint,
        config.config_file,
        config.vocab_file,
        config.dvae_checkpoint,
        config.mel_stats_file,
    ]

    # Check if all files already exist locally
    all_present = all(
        os.path.isfile(os.path.join(config.base_model_dir, f))
        for f in required_files
    )

    if all_present:
        logger.info(f"Base model already present at: {config.base_model_dir}")
        return config.base_model_dir

    logger.info(f"Downloading base model from HuggingFace: {config.hf_repo_id}")
    os.makedirs(config.base_model_dir, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=config.hf_repo_id,
        local_dir=config.base_model_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", "*.gitattributes"],
    )

    logger.info(f"Model downloaded to: {local_dir}")
    return local_dir


# ─── XTTS Model Loading ───────────────────────────────────────────────────────

def load_xtts_model(
    config: TrainingConfig,
    logger: logging.Logger,
    checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, object]:
    """
    Load the XTTS v2 model using Coqui TTS internal APIs.

    Args:
        config:           TrainingConfig
        logger:           Logger instance
        checkpoint_path:  If provided, load weights from this .pth file
                          instead of the base model.

    Returns:
        (model, xtts_config) — the XTTS model and its config object
    """
    # ── Ensure numpy compatibility before importing TTS ───────────────────────
    _ensure_numpy_compat()

    # Import Coqui TTS internals
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as e:
        raise ImportError(
            "Coqui TTS is not installed. Run: pip install TTS\n"
            f"Original error: {e}"
        )

    # ── Load XTTS config ──────────────────────────────────────────────────────
    logger.info(f"Loading XTTS config from: {config.config_path}")
    xtts_config = XttsConfig()
    xtts_config.load_json(config.config_path)

    # Override paths to point to our local files
    xtts_config.model_args.dvae_checkpoint = config.dvae_path
    xtts_config.model_args.tokenizer_file = config.vocab_path

    # ── Instantiate model ─────────────────────────────────────────────────────
    logger.info("Instantiating XTTS model...")
    model = Xtts.init_from_config(xtts_config)

    # ── Load weights ──────────────────────────────────────────────────────────
    weights_path = checkpoint_path if checkpoint_path else config.model_path
    logger.info(f"Loading weights from: {weights_path}")

    model.load_checkpoint(
        xtts_config,
        checkpoint_path=weights_path,
        vocab_path=config.vocab_path,
        eval=False,
        use_deepspeed=False,
    )

    # ── Load mel stats ────────────────────────────────────────────────────────
    if os.path.isfile(config.mel_stats_path):
        logger.info(f"Loading mel stats from: {config.mel_stats_path}")
        mel_stats = torch.load(config.mel_stats_path, map_location="cpu")
        if hasattr(model, "mel_stats"):
            model.mel_stats = mel_stats

    # ── Move to device ────────────────────────────────────────────────────────
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model moved to: {device}")

    return model, xtts_config


# ─── Freeze / Unfreeze Layers ─────────────────────────────────────────────────

def configure_trainable_params(
    model: nn.Module,
    config: TrainingConfig,
    logger: logging.Logger,
) -> nn.Module:
    """
    Freeze encoder layers and keep decoder + speaker components trainable.
    This reduces VRAM usage and speeds up training.
    """
    if not config.freeze_encoder:
        # Train everything
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Full fine-tune: all {total:,} parameters trainable")
        return model

    # ── Print actual top-level module names for debugging ─────────────────────
    top_level = [name for name, _ in model.named_children()]
    logger.debug(f"Top-level modules: {top_level}")

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Components to train — covers both old and new Coqui TTS naming conventions
    trainable_keywords = [
        # Decoder / autoregressive
        "gpt",
        "mel_head",
        "final_norm",
        # Vocoder
        "hifigan_decoder",
        "hifi_decoder",
        "vocoder",
        # Speaker conditioning
        "speaker_encoder",
        "conditioning_encoder",
        "cond_encoder",
        "speaker_embedding",
        "speaker_emb",
        # Diffusion (XTTS v2 uses diffusion for mel refinement)
        "diffusion",
        "diffusion_decoder",
    ]

    if not config.freeze_encoder:
        trainable_keywords.append("text_encoder")

    total_params     = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True
            trainable_params += param.numel()

    # Safety: if nothing was unfrozen (keyword mismatch), train everything
    if trainable_params == 0:
        logger.warning(
            "No parameters matched trainable keywords — "
            "falling back to full fine-tune. "
            f"Top-level modules: {top_level}"
        )
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters())

    pct = 100.0 * trainable_params / max(total_params, 1)
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({pct:.1f}%)"
    )

    return model


# ─── Gradient Checkpointing ───────────────────────────────────────────────────

def enable_gradient_checkpointing(model: nn.Module, logger: logging.Logger):
    """
    Enable gradient checkpointing on transformer/GPT layers to save VRAM.
    Trades compute for memory.

    XTTS uses a custom GPT2Model that does NOT have standard HuggingFace
    embeddings (wte/wpe), so we must catch AttributeError from
    enable_input_require_grads() and fall back to a manual hook.
    """
    enabled = False

    # Try to enable on GPT component first (preferred)
    if hasattr(model, "gpt") and hasattr(model.gpt, "gradient_checkpointing_enable"):
        try:
            model.gpt.gradient_checkpointing_enable()
            enabled = True
            logger.info("Gradient checkpointing enabled on GPT component")
        except (AttributeError, NotImplementedError) as e:
            logger.debug(f"GPT gradient_checkpointing_enable failed: {e}")

    # Generic fallback: walk all submodules
    if not enabled:
        for name, module in model.named_modules():
            if not hasattr(module, "gradient_checkpointing_enable"):
                continue
            try:
                module.gradient_checkpointing_enable()
                logger.info(f"Gradient checkpointing enabled on: {name}")
                enabled = True
                break
            except (AttributeError, NotImplementedError) as e:
                # XTTS GPT2Model lacks wte — skip silently
                logger.debug(f"Skipping {name}: {e}")
                continue

    # Last resort: manually set the flag on any module that has it
    if not enabled:
        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                enabled = True
                logger.info(f"Gradient checkpointing flag set on: {name}")
                break

    if enabled:
        logger.info("✅ Gradient checkpointing active")
    else:
        logger.warning(
            "Could not enable gradient checkpointing — "
            "model architecture does not support it. Training will use more VRAM."
        )


# ─── LoRA Support (Bonus) ─────────────────────────────────────────────────────

def apply_lora(
    model: nn.Module,
    config: TrainingConfig,
    logger: logging.Logger,
) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to the XTTS model's attention layers.
    Requires the `peft` library: pip install peft
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.warning(
            "peft library not found. LoRA disabled. "
            "Install with: pip install peft"
        )
        return model

    logger.info(
        f"Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}, "
        f"targets={config.lora_target_modules}"
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        # XTTS is a seq2seq-like model; use FEATURE_EXTRACTION task type
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA applied successfully")
    except Exception as e:
        logger.warning(f"LoRA application failed: {e}. Continuing without LoRA.")

    return model


# ─── Speaker Embedding Extraction ─────────────────────────────────────────────

def extract_speaker_embedding(
    model: nn.Module,
    xtts_config: object,
    audio_path: str,
    device: torch.device,
    logger: logging.Logger,
) -> Optional[torch.Tensor]:
    """
    Extract a speaker embedding from a reference audio file using XTTS.

    Returns:
        Speaker embedding tensor of shape [1, D] or None on failure.
    """
    try:
        import torchaudio
        import torchaudio.transforms as T

        # Load and preprocess reference audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 22050:
            resampler = T.Resample(orig_freq=sr, new_freq=22050)
            waveform = resampler(waveform)

        waveform = waveform.to(device)

        model.eval()
        with torch.no_grad():
            # Use XTTS's built-in speaker encoder
            if hasattr(model, "get_conditioning_latents"):
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[audio_path]
                )
                logger.info(
                    f"Speaker embedding extracted: shape={speaker_embedding.shape}"
                )
                return speaker_embedding
            else:
                logger.warning("Model does not have get_conditioning_latents method")
                return None

    except Exception as e:
        logger.error(f"Failed to extract speaker embedding: {e}")
        return None


# ─── Checkpoint Save / Load ───────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    patch_idx: int,
    step: int,
    loss: float,
    config: TrainingConfig,
    logger: logging.Logger,
    is_best: bool = False,
):
    """Save model + optimizer state to a checkpoint file."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Get the actual model (unwrap LoRA/DataParallel if needed)
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "patch_idx": patch_idx,
        "step": step,
        "loss": loss,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.__dict__,
    }

    # Patch checkpoint
    ckpt_path = os.path.join(config.checkpoint_dir, f"patch_{patch_idx:04d}.pth")
    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path} (loss={loss:.4f})")

    # Best model checkpoint
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        logger.info(f"Best model updated: {best_path}")

    # Optional zip
    if config.zip_checkpoints:
        from .utils import zip_checkpoint
        zip_checkpoint(ckpt_path, logger)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).
    Returns the checkpoint metadata dict.
    """
    if logger:
        logger.info(f"Loading checkpoint: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)

    # Unwrap model if needed
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(state["model_state_dict"], strict=False)

    if optimizer and "optimizer_state_dict" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            if logger:
                logger.warning(f"Could not load optimizer state: {e}")

    if scheduler and state.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            if logger:
                logger.warning(f"Could not load scheduler state: {e}")

    meta = {
        "patch_idx": state.get("patch_idx", 0),
        "step": state.get("step", 0),
        "loss": state.get("loss", float("inf")),
    }

    if logger:
        logger.info(
            f"Checkpoint loaded: patch={meta['patch_idx']}, "
            f"step={meta['step']}, loss={meta['loss']:.4f}"
        )

    return meta
