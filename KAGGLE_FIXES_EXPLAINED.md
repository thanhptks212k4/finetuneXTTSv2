# Kaggle XTTS Training - Critical Fixes Explained

## 🔧 All 4 Critical Fixes Applied

### ✅ Fix #1: W&B Hanging Issue (MOST CRITICAL)

**Problem:** Script hangs indefinitely waiting for Weights & Biases API key on Kaggle.

**Solution:**
```python
# MUST be at the very top, BEFORE any imports
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
```

**Why it works:** Disables W&B before TTS/torch libraries try to initialize it.

**Location in script:** Lines 10-12 (before all other imports)

---

### ✅ Fix #2: DeepSpeed Missing Dependency

**Problem:** Kaggle T4 GPU hits OOM during backward pass without DeepSpeed optimization.

**Solution:**
```python
# In STEP 2, add deepspeed to pip install
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q', 
     'huggingface_hub', 'librosa', 'soundfile', 'torchaudio', 'deepspeed'],
    stdout=subprocess.DEVNULL,
)
```

**Why it works:** DeepSpeed provides memory-efficient training optimizations for large models.

**Location in script:** Lines 68-72

---

### ✅ Fix #3: Better Path Error Handling

**Problem:** Cryptic errors when Kaggle datasets aren't mounted correctly.

**Solution:**
```python
# Detailed validation with helpful error messages
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
    raise FileNotFoundError(f'Missing {len(missing_paths)} required path(s)')
```

**Why it works:** Provides clear, actionable error messages instead of generic FileNotFoundError.

**Location in script:** Lines 91-115

---

### ✅ Fix #4: Explicit Garbage Collection

**Problem:** Residual VRAM from model loading causes OOM during training initialization.

**Solution:**
```python
# CRITICAL: Clear memory before training
print('\n🧹 Clearing memory before training...')
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
log_gpu_memory(logger, 'after garbage collection')
print('✅ Memory cleared')
```

**Why it works:** 
- `gc.collect()` - Frees Python objects
- `torch.cuda.empty_cache()` - Releases cached GPU memory
- `torch.cuda.synchronize()` - Ensures all GPU operations complete

**Location in script:** Lines 223-230

---

## 📊 Additional Optimizations

### Memory Optimizations for Kaggle T4 (16GB VRAM)

```python
config = TrainingConfig(
    patch_size = 2000,              # Smaller patches (was 3000)
    batch_size = 2,                 # Small batch for T4
    grad_accum_steps = 8,           # Effective batch = 16
    num_workers = 1,                # Single worker
    use_fp16 = True,                # Half precision
    gradient_checkpointing = True,  # Trade compute for memory
    freeze_encoder = True,          # Only train decoder
)
```

### Monitoring Improvements

```python
eval_every_n_steps = 500,       # More frequent eval
save_every_n_steps = 500,       # More frequent saves
```

This allows you to:
- Monitor training progress more closely
- Recover from interruptions more easily
- Catch issues earlier

---

## 🚀 How to Use

### Option 1: Run as Python Script

```bash
# In Kaggle notebook cell:
!python /kaggle/working/kaggle_notebook_auto_fixed.py
```

### Option 2: Copy-Paste into Notebook

1. Create new code cell in Kaggle notebook
2. Copy entire content of `kaggle_notebook_auto_fixed.py`
3. Paste and run

### Option 3: Upload as Notebook

1. Convert to `.ipynb` format
2. Upload to Kaggle
3. Run all cells

---

## 🔍 Verification Checklist

Before running, verify:

- [ ] **Datasets added** in Kaggle settings:
  - `tinthnhphm21022004/data-speech-to-text`
  - `thanhphamtien2102224/weight-phowhisper`

- [ ] **GPU enabled**: Settings → Accelerator → GPU T4 x1

- [ ] **Internet ON**: Settings → Internet → ON

- [ ] **Persistence**: Settings → Persistence → Files only

---

## 📈 Expected Behavior

### Successful Run

```
[STEP 0] Checking GPU...
✅ GPU: Tesla T4
✅ VRAM: 15.8 GB

[STEP 1] Cloning repository...
✅ Repository cloned

[STEP 2] Installing dependencies (including DeepSpeed)...
✅ All dependencies installed (including DeepSpeed)

[STEP 3] Verifying installations...
✅ numpy: 1.24.3
✅ TTS: 0.22.0
✅ PyTorch: 2.1.0
✅ DeepSpeed: 0.12.3

[STEP 4] Setting up paths...
✅ All dataset paths exist
✅ Train: 45,234 valid samples
✅ Test: 5,678 valid samples

[STEP 5] Configuring training...
✅ Config ready
   Effective batch size: 16

[STEP 6] Loading dataset...
✅ Train: 45,234 valid samples
✅ Val: 5,678 valid samples

[STEP 7] Downloading base model...
✅ Base model ready

[STEP 8] Loading XTTS model...
✅ Speaker embedding: torch.Size([1, 512, 1])
🧹 Clearing memory before training...
✅ Memory cleared

[STEP 9] Starting training...
[INFO] step=50 | total_loss=2.5432 | mel_loss=1.8234
[INFO] step=100 | total_loss=2.3456 | mel_loss=1.6543
...
```

### If Datasets Missing

```
❌ DATASET PATHS NOT FOUND

Missing paths:
  ❌ /kaggle/input/datasets/tinthnhphm21022004/...

📋 How to fix:
1. Go to Kaggle notebook settings
2. Click "Add Data" → "Datasets"
3. Add these datasets:
   - tinthnhphm21022004/data-speech-to-text
   - thanhphamtien2102224/weight-phowhisper
4. Re-run the notebook
```

---

## ⏱️ Expected Runtime

- **Total time**: 2-3 hours on Kaggle T4
- **Per patch**: ~15-20 minutes
- **Total patches**: ~6-8 (depends on dataset size)

---

## 💾 Output Files

After successful run:

```
/kaggle/working/output/
├── checkpoints/
│   ├── patch_0001.pth
│   ├── patch_0002.pth
│   └── best_model.pth
├── samples/
│   ├── step_000500_sample_00.wav
│   ├── step_001000_sample_00.wav
│   └── ...
└── logs/
    └── training.log
```

---

## 🐛 Troubleshooting

### Issue: Still hangs at start

**Cause:** W&B environment variables not set early enough

**Fix:** Ensure lines 10-12 are at the VERY TOP, before any other imports

### Issue: OOM during training

**Cause:** Batch size too large or DeepSpeed not installed

**Fix:** 
1. Verify DeepSpeed installed: `pip show deepspeed`
2. Reduce batch_size to 1: `batch_size = 1`
3. Increase grad_accum_steps: `grad_accum_steps = 16`

### Issue: "No valid training samples"

**Cause:** Audio files don't match manifest paths

**Fix:** Check AUDIO_ROOT path matches your dataset structure

### Issue: Loss stays at 0.0000

**Cause:** GPT forward not working (known issue with some XTTS versions)

**Fix:** Script will automatically use Variant B fallback. You'll see:
```
[WARNING] Using Variant B: mel reconstruction via param mean
[WARNING] This is a FALLBACK - GPT forward is not working
```

Training will continue but with limited effectiveness.

---

## 📞 Support

If issues persist:

1. Check `/kaggle/working/output/logs/training.log`
2. Check `/kaggle/working/output/error_debug.txt` (if training failed)
3. Review the error messages - they're designed to be helpful!

---

## ✅ Success Criteria

Your training is working correctly if:

- ✅ No W&B hanging
- ✅ DeepSpeed installed and detected
- ✅ Loss decreases over time (not stuck at 0.0000)
- ✅ Checkpoints save successfully
- ✅ Audio samples generate
- ✅ No OOM errors

---

**Version:** 1.0 (May 2026)  
**Tested on:** Kaggle T4 GPU (16GB VRAM)  
**Status:** Production-ready ✅
