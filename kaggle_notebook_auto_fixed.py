#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS v2 Vietnamese Fine-Tuning - FULLY FIXED FOR KAGGLE
Expert-level fixes for W&B hanging, OOM, and path issues
"""

# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX #1: Disable W&B BEFORE any imports
# ══════════════════════════════════════════════════════════════════════════════
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import sys
import json
import shutil
import subprocess
import importlib
import gc
from pathlib import Path

print('='*80)
print('🚀 XTTS v2 Vietnamese Fine-Tuning - FULLY FIXED VERSION')
print('='*80)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Check GPU
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 0] Checking GPU...')
import torch

print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  No GPU detected! Training will be very slow.')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Clone repository
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 1] Cloning repository...')
REPO_DIR = '/kaggle/working/finetuneXTTSv2'

if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
    print('🗑️  Removed old repository')

stale = [k for k in sys.modules if k.startswith('xtts_finetune')]
for k in stale:
    del sys.modules[k]
if stale:
    print(f'🗑️  Cleared {len(stale)} cached modules')

subprocess.check_call(
    ['git', 'clone', 'https://github.com/thanhptks212k4/finetuneXTTSv2.git', REPO_DIR],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f'✅ Repository cloned to {REPO_DIR}')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
importlib.invalidate_caches()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Install dependencies (CRITICAL FIX #2: Add DeepSpeed)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 2] Installing dependencies (including DeepSpeed)...')
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'],
    stdout=subprocess.DEVNULL,
)
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/idiap/coqui-ai-TTS.git'],
    stdout=subprocess.DEVNULL,
)
# CRITICAL: Add deepspeed to prevent OOM during backward pass
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'librosa', 'soundfile', 'torchaudio', 'deepspeed'],
    stdout=subprocess.DEVNULL,
)
print('✅ All dependencies installed (including DeepSpeed)')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Verify installations
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 3] Verifying installations...')
import numpy as np
import TTS as _tts_pkg
print(f'✅ numpy: {np.__version__}')
print(f'✅ TTS: {_tts_pkg.__version__}')
print(f'✅ PyTorch: {torch.__version__}')

try:
    import deepspeed
    print(f'✅ DeepSpeed: {deepspeed.__version__}')
except ImportError:
    print('⚠️  DeepSpeed not found - may hit OOM errors')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Setup paths and fix manifests (CRITICAL FIX #3: Better error handling)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 4] Setting up paths and fixing manifests...')

AUDIO_ROOT  = '/kaggle/input/datasets/tinthnhphm21022004/data-speech-to-text/data_kagglee_wav/data_kagglee_wav'
TRAIN_JSONL = '/kaggle/input/datasets/thanhphamtien2102224/weight-phowhisper/train_full_manifest.jsonl'
TEST_JSONL  = '/kaggle/input/datasets/thanhphamtien2102224/weight-phowhisper/test_manifest.jsonl'

# CRITICAL FIX #3: Detailed path validation with helpful error messages
missing_paths = []
for p in [AUDIO_ROOT, TRAIN_JSONL, TEST_JSONL]:
    if not os.path.exists(p):
        missing_paths.append(p)

if missing_paths:
    print('\n' + '='*80)
    print('❌ DATASET PATHS NOT FOUND')
    print('='*80)
    print('\nMissing paths:')
    for p in missing_paths:
        print(f'  ❌ {p}')
    print('\n📋 How to fix:')
    print('1. Go to Kaggle notebook settings')
    print('2. Click "Add Data" → "Datasets"')
    print('3. Add these datasets:')
    print('   - tinthnhphm21022004/data-speech-to-text')
    print('   - thanhphamtien2102224/weight-phowhisper')
    print('4. Re-run the notebook')
    print('\n💡 Tip: Check if dataset slugs have changed on Kaggle')
    print('='*80)
    raise FileNotFoundError(f'Missing {len(missing_paths)} required path(s)')

print('✅ All dataset paths exist')

FIXED_TRAIN = '/kaggle/working/train_manifest.jsonl'
FIXED_TEST  = '/kaggle/working/test_manifest.jsonl'

def fix_manifest(src, dst, audio_root):
    """Fix manifest paths and validate audio files exist"""
    ok, missing = 0, 0
    with open(src, 'r', encoding='utf-8') as fin, \
         open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = json.loads(line.strip())
            # Fix audio path
            audio_file = os.path.basename(obj['audio'])
            obj['audio'] = os.path.join(audio_root, audio_file)
            
            if os.path.exists(obj['audio']):
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
                ok += 1
            else:
                missing += 1
    return ok, missing

train_ok, train_miss = fix_manifest(TRAIN_JSONL, FIXED_TRAIN, AUDIO_ROOT)
test_ok, test_miss = fix_manifest(TEST_JSONL, FIXED_TEST, AUDIO_ROOT)
print(f'✅ Train: {train_ok:,} valid samples ({train_miss} missing)')
print(f'✅ Test: {test_ok:,} valid samples ({test_miss} missing)')

# Find reference audio
import glob
wav_files = glob.glob(os.path.join(AUDIO_ROOT, '*.wav'))
if not wav_files:
    raise FileNotFoundError(f'❌ No .wav files found in {AUDIO_ROOT}')
REFERENCE_WAV = wav_files[0]

import soundfile as sf
info = sf.info(REFERENCE_WAV)
dur = info.frames / info.samplerate
print(f'✅ Reference audio: {os.path.basename(REFERENCE_WAV)}')
print(f'   Duration: {dur:.2f}s | SR: {info.samplerate} Hz')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Configure training (OPTIMIZED FOR KAGGLE T4)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 5] Configuring training...')

from xtts_finetune.config import TrainingConfig
from xtts_finetune.utils import get_logger, set_seed

WORKING    = '/kaggle/working'
OUTPUT_DIR = f'{WORKING}/output'

config = TrainingConfig(
    hf_repo_id     = 'coqui/XTTS-v2',
    base_model_dir = f'{WORKING}/base_model',
    train_manifest  = FIXED_TRAIN,
    val_manifest    = FIXED_TEST,
    reference_audio = REFERENCE_WAV,
    audio_root_remap = None,
    
    # KAGGLE T4 OPTIMIZATIONS
    patch_size = 2000,              # Smaller patches (was 3000)
    batch_size = 2,                 # Small batch for T4
    grad_accum_steps = 8,           # Effective batch = 16
    eval_every_n_steps = 500,       # More frequent eval for monitoring
    save_every_n_steps = 500,       # More frequent saves
    num_workers = 1,                # Single worker to save memory
    zip_checkpoints = False,        # Don't zip (saves time)
    
    output_dir     = OUTPUT_DIR,
    checkpoint_dir = f'{OUTPUT_DIR}/checkpoints',
    sample_dir     = f'{OUTPUT_DIR}/samples',
    log_dir        = f'{OUTPUT_DIR}/logs',
    
    learning_rate    = 2e-5,
    epochs_per_patch = 1,
    use_fp16               = True,
    gradient_checkpointing = True,
    freeze_encoder         = True,
    speaker_mode = 'single',
    seed = 42,
)

logger = get_logger('kaggle_auto', config.log_dir)
set_seed(42)

print(f'✅ Config ready')
print(f'   Effective batch size: {config.batch_size * config.grad_accum_steps}')
print(f'   Patch size: {config.patch_size:,} samples')
print(f'   Output dir: {OUTPUT_DIR}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Load and validate dataset
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 6] Loading and validating dataset...')

from xtts_finetune.dataset import load_manifest, validate_and_filter

train_raw = load_manifest(config.train_manifest, logger, config.audio_root_remap)
val_raw = load_manifest(config.val_manifest, logger, config.audio_root_remap)

print(f'Loaded {len(train_raw):,} train samples (raw)')
print(f'Loaded {len(val_raw):,} val samples (raw)')

train_samples = validate_and_filter(train_raw, config, logger)
val_samples = validate_and_filter(val_raw, config, logger)

print(f'✅ Train: {len(train_samples):,} valid samples')
print(f'✅ Val: {len(val_samples):,} valid samples')

if len(train_samples) == 0:
    raise RuntimeError('❌ No valid training samples after validation!')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Download base model
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 7] Downloading base model from HuggingFace...')

from xtts_finetune.model_loader import download_base_model

download_base_model(config, logger)
print('✅ Base model ready')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Load model and configure (CRITICAL FIX #4: Explicit garbage collection)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 8] Loading XTTS model...')

from xtts_finetune.model_loader import (
    load_xtts_model,
    configure_trainable_params,
    enable_gradient_checkpointing,
    extract_speaker_embedding,
)
from xtts_finetune.utils import log_gpu_memory, free_memory

model, xtts_config = load_xtts_model(config, logger)
model = configure_trainable_params(model, config, logger)
enable_gradient_checkpointing(model, logger)
log_gpu_memory(logger, 'after model load')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
speaker_embedding = extract_speaker_embedding(
    model, xtts_config, config.reference_audio, device, logger
)

if speaker_embedding is not None:
    print(f'✅ Speaker embedding: {speaker_embedding.shape}')
else:
    print('⚠️  Using default speaker embedding')

# CRITICAL FIX #4: Explicit garbage collection before training
print('\n🧹 Clearing memory before training...')
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
log_gpu_memory(logger, 'after garbage collection')
print('✅ Memory cleared')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Train
# ══════════════════════════════════════════════════════════════════════════════
print('\n[STEP 9] Starting training...')
print('='*80)

from xtts_finetune.trainer import XTTSTrainer

trainer = XTTSTrainer(
    model             = model,
    xtts_config       = xtts_config,
    config            = config,
    speaker_embedding = speaker_embedding,
)

try:
    final_metrics = trainer.train(train_samples, val_samples)
    
    print('\n' + '='*80)
    print('🎉 TRAINING COMPLETED SUCCESSFULLY!')
    print('='*80)
    print(f'Best validation loss: {trainer.best_val_loss:.4f}')
    print(f'Total steps: {trainer.global_step}')
    print(f'\n📂 Output locations:')
    print(f'   Checkpoints: {config.checkpoint_dir}')
    print(f'   Samples: {config.sample_dir}')
    print(f'   Logs: {config.log_dir}')
    
except KeyboardInterrupt:
    print('\n' + '='*80)
    print('⚠️  TRAINING INTERRUPTED BY USER')
    print('='*80)
    print(f'Partial results saved to: {OUTPUT_DIR}')
    
except Exception as e:
    print('\n' + '='*80)
    print('❌ TRAINING FAILED')
    print('='*80)
    print(f'Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    
    # Save debug info
    debug_file = f'{OUTPUT_DIR}/error_debug.txt'
    with open(debug_file, 'w') as f:
        f.write(f'Error: {type(e).__name__}: {e}\n\n')
        f.write('Traceback:\n')
        traceback.print_exc(file=f)
    print(f'\n💾 Debug info saved to: {debug_file}')
    raise

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*80)
print('✅ ALL STEPS COMPLETED!')
print('='*80)
print('\n📊 Training Summary:')
print(f'   Total samples trained: {len(train_samples):,}')
print(f'   Validation samples: {len(val_samples):,}')
print(f'   Total steps: {trainer.global_step}')
print(f'   Best val loss: {trainer.best_val_loss:.4f}')
print('\n📂 Output locations:')
print(f'   Checkpoints: {config.checkpoint_dir}')
print(f'   Samples: {config.sample_dir}')
print(f'   Logs: {config.log_dir}')
print('\n🎉 Training pipeline completed successfully!')
print('='*80)
