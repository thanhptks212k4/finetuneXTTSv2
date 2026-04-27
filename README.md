# XTTS v2 Vietnamese Fine-Tuning Pipeline

Production-ready pipeline to fine-tune **XTTS v2** on a custom Vietnamese dataset using Coqui TTS internal APIs.

## Features

| Feature | Details |
|---|---|
| Base model | `anhnh2002/vnTTS` (HuggingFace) |
| Framework | Coqui TTS + PyTorch (no HF Trainer) |
| Training | Patch-based (5000 samples/patch) |
| Memory | fp16, gradient checkpointing, `empty_cache()` |
| Speaker | Single-speaker & multi-speaker modes |
| Losses | Mel reconstruction + duration + speaker consistency |
| Evaluation | MCD (Mel Cepstral Distortion) per epoch |
| Checkpoints | Per-patch + best model + optional zip |
| Bonus | LoRA support, Vietnamese text normalization |
| Target | Kaggle T4 GPU (16 GB VRAM) |

---

## Project Structure

```
xtts_finetune/
├── __init__.py          # Package exports
├── config.py            # All hyperparameters and paths
├── dataset.py           # Manifest loading, validation, patch splitting, DataLoader
├── model_loader.py      # HF download, XTTS load, freeze/unfreeze, LoRA, checkpoints
├── trainer.py           # Training loop, loss functions, evaluation, sample generation
├── inference.py         # Post-training inference CLI
├── prepare_data.py      # Build JSONL manifests from raw audio directories
└── utils.py             # Logger, seed, memory, Vietnamese normalization, MCD

requirements.txt         # Python dependencies
kaggle_notebook.ipynb    # Ready-to-run Kaggle notebook
README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install TTS>=0.22.0 huggingface_hub torchaudio librosa soundfile
# Optional LoRA:
pip install peft
```

### 2. Prepare your dataset

**Option A — Flat layout** (each `.wav` has a matching `.txt`):
```
raw_data/
    utt001.wav
    utt001.txt
    utt002.wav
    utt002.txt
```

**Option B — LJSpeech layout**:
```
raw_data/
    wavs/
        utt001.wav
    metadata.csv   # format: filename|text
```

Build manifests:
```bash
python -m xtts_finetune.prepare_data \
    --input_dir ./raw_data \
    --layout flat \          # or ljspeech
    --output_dir ./data \
    --val_ratio 0.05
```

This creates `data/train.jsonl` and `data/val.jsonl`.

### 3. Add a reference audio

Copy a clean 5–10 second recording of your target speaker to `./data/reference.wav`.

### 4. Train

```bash
python -m xtts_finetune.main \
    --train_manifest ./data/train.jsonl \
    --val_manifest   ./data/val.jsonl \
    --reference_audio ./data/reference.wav \
    --batch_size 4 \
    --epochs_per_patch 1
```

### 5. Resume training

```bash
python -m xtts_finetune.main --resume_patch 3
```

### 6. Inference

```bash
python -m xtts_finetune.main --inference_only \
    --text "Xin chào, đây là giọng nói tổng hợp." \
    --reference_audio ./data/reference.wav \
    --inference_output ./output/result.wav
```

Or use the inference module directly:
```bash
python -m xtts_finetune.inference \
    --text "Xin chào thế giới" \
    --reference_audio ./data/reference.wav \
    --output ./output/result.wav
```

---

## Manifest Format

Each line in the JSONL manifest:
```json
{"audio": "path/to/audio.wav", "text": "transcription text"}
```

For multi-speaker datasets, add a `speaker_id` field:
```json
{"audio": "path/to/audio.wav", "text": "transcription", "speaker_id": "speaker_01"}
```

---

## Configuration

All settings live in `config.py`. Key parameters:

```python
config = TrainingConfig(
    # Model
    hf_repo_id="anhnh2002/vnTTS",
    base_model_dir="./base_model",

    # Data
    train_manifest="./data/train.jsonl",
    val_manifest="./data/val.jsonl",
    patch_size=5000,           # samples per patch

    # Training
    batch_size=4,
    grad_accum_steps=4,        # effective batch = 16
    learning_rate=2e-5,
    epochs_per_patch=1,

    # Memory
    use_fp16=True,
    gradient_checkpointing=True,
    freeze_encoder=True,       # only train decoder + speaker

    # Speaker
    speaker_mode="single",     # "single" | "multi"
    reference_audio="./data/reference.wav",

    # LoRA (bonus)
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
)
```

---

## Kaggle T4 Recommended Settings

```python
config = TrainingConfig(
    batch_size=2,
    grad_accum_steps=8,    # effective batch = 16
    use_fp16=True,
    gradient_checkpointing=True,
    freeze_encoder=True,
    zip_checkpoints=True,  # save disk space
    num_workers=2,
)
```

---

## Memory Optimization

The pipeline uses several strategies to stay within T4 VRAM limits:

1. **Patch-based loading** — only 5000 samples in RAM at a time
2. **fp16 mixed precision** — halves activation memory
3. **Gradient checkpointing** — recomputes activations instead of storing them
4. **Encoder freezing** — only decoder + speaker layers have gradients
5. **`torch.cuda.empty_cache()` + `gc.collect()`** — called after each patch
6. **Small batch size** — 2–4 with gradient accumulation

---

## Loss Functions

| Loss | Weight | Description |
|---|---|---|
| `mel_loss` | 1.0 | L1 between predicted and target mel spectrograms |
| `duration_loss` | 0.1 | MSE on log-durations |
| `speaker_loss` | 0.1 | 1 − cosine similarity of speaker embeddings |

---

## Evaluation Metric

**MCD (Mel Cepstral Distortion)** — measures spectral distance between synthesized and reference audio.
- Lower is better
- Typical range: 5–10 dB
- Good fine-tuning: < 7 dB

---

## LoRA (Bonus)

Enable LoRA for parameter-efficient fine-tuning:

```bash
pip install peft
python -m xtts_finetune.main --use_lora
```

LoRA targets attention projection layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`) with rank 16.

---

## License

This pipeline is provided for research and educational use.
The base model (`anhnh2002/vnTTS`) is subject to its own license on HuggingFace.
Coqui TTS is licensed under the Mozilla Public License 2.0.
