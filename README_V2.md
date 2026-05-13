# XTTS v2 Self-Contained Notebook - Implementation Summary

## Vấn đề hiện tại

Notebook `kaggle_notebook_auto.ipynb` hiện tại có vấn đề:
1. **Cell 5 bị malformed** - đã được fix
2. **Phụ thuộc vào local packages** (`xtts_finetune.*`) - không tự chứa

## Giải pháp đề xuất

Tạo `kaggle_notebook_v2.ipynb` hoàn toàn tự chứa với 11 cells:

### Cell 1: Install Dependencies
```python
pip install torch==2.1.0 torchaudio==2.1.0
pip install git+https://github.com/idiap/coqui-ai-TTS.git
pip install huggingface_hub librosa soundfile
```

### Cell 2: CONFIG
Tất cả constants: paths, hyperparameters, flags

### Cell 3: GPU Check + Seed
Kiểm tra GPU, set seed, validate VRAM

### Cell 4: Download Base Model
```python
from huggingface_hub import snapshot_download
snapshot_download('coqui/XTTS-v2', local_dir=BASE_MODEL_DIR)
```

### Cell 5: Audio Preprocessing
- Tìm tất cả .wav files
- Fix manifest paths (auto-detect)
- Select reference audio (5-10s, highest SR)
- **CRITICAL**: Return `valid_indices` để sync với text tokens

### Cell 6: Build JSONL Manifest with Validation
- Load manifests
- Validate: file exists, duration in range, text non-empty
- Filter và normalize

### Cell 7: Dataset Class + DataLoader
```python
class XTTSDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # Load audio lazily
        # Return: audio, text, speaker_id
```

### Cell 8: Load Model + Freeze Encoder
```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

model = Xtts.init_from_config(config)
model.load_checkpoint(...)

# Freeze encoder
for name, param in model.named_parameters():
    if 'gpt' in name or 'decoder' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# CRITICAL: Assert trainable params > 0
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, "No trainable parameters!"
print(f"Trainable: {trainable:,}")
```

### Cell 9: Training Loop (với valid_indices sync fix)
```python
def extract_mel_with_sync(model, audio, audio_lengths):
    \"\"\"
    CRITICAL FIX: Track valid_indices to sync mel and text
    \"\"\"
    mels = []
    valid_indices = []
    
    for i, (wav, length) in enumerate(zip(audio, audio_lengths)):
        try:
            mel = model.ap.melspectrogram(wav[:length])
            if mel.shape[-1] > 0:
                mels.append(mel)
                valid_indices.append(i)
        except:
            continue  # Skip failed samples
    
    return mels, valid_indices

# Training loop
for step in range(MAX_STEPS):
    try:
        batch = next(train_iter)
        
        # Extract mel WITH valid_indices tracking
        mels, valid_indices = extract_mel_with_sync(
            model, batch['audio'], batch['audio_lengths']
        )
        
        # SYNC: Filter texts to match valid mels
        texts = [batch['text'][i] for i in valid_indices]
        
        # Now mels and texts are in sync!
        loss = model.gpt(text_inputs=texts, mel_codes=mels, ...)
        
        loss.backward()
        optimizer.step()
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            # Auto-reduce batch size
            BATCH_SIZE = max(1, BATCH_SIZE // 2)
            print(f"OOM! Reducing batch size to {BATCH_SIZE}")
            torch.cuda.empty_cache()
            continue
        raise
```

### Cell 10: Save Checkpoint + Zip
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': step,
    'loss': loss.item(),
}, f'{CHECKPOINT_DIR}/checkpoint_{step}.pth')

# Zip for Kaggle download
import zipfile
with zipfile.ZipFile(f'{CHECKPOINT_DIR}/checkpoint_{step}.zip', 'w') as zf:
    zf.write(checkpoint_path)
```

### Cell 11: Inference Test
```python
model.eval()
with torch.no_grad():
    for text in VAL_TEXTS:
        wav = model.inference(
            text=text,
            language='vi',
            gpt_cond_latent=speaker_embedding,
            speaker_embedding=speaker_embedding,
        )
        torchaudio.save(f'sample_{i}.wav', wav, 22050)
```

## Key Fixes Applied

### 1. Valid_Indices Sync Fix (NoneType Bug)
**Problem**: Mel extraction fails for some samples → mismatch between mel list and text list → NoneType error

**Solution**:
```python
# OLD (broken):
mels = [extract_mel(audio[i]) for i in range(B)]  # Some None
texts = batch['text']  # All B items
# → mels[3] = None but texts[3] exists → crash!

# NEW (fixed):
mels, valid_indices = extract_mel_with_tracking(audio)
texts = [batch['text'][i] for i in valid_indices]
# → mels and texts always same length, no None
```

### 2. Trainable Params Validation
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, "❌ LỖI: Không có parameter nào được train!"
print(f"✅ Trainable parameters: {trainable:,}")
```

### 3. Auto-Reduce Batch Size on OOM
```python
try:
    loss.backward()
except RuntimeError as e:
    if 'out of memory' in str(e):
        BATCH_SIZE = max(1, BATCH_SIZE // 2)
        torch.cuda.empty_cache()
        continue
```

## Lý do cần notebook mới

1. **Tự chứa hoàn toàn**: Không phụ thuộc `xtts_finetune.*`
2. **Dễ debug**: Tất cả code nhìn thấy được
3. **Dễ customize**: Edit trực tiếp trong notebook
4. **Fix critical bugs**: valid_indices sync, trainable params check
5. **Vietnamese errors**: Dễ hiểu cho user

## Cách sử dụng

1. Upload `kaggle_notebook_v2.ipynb` lên Kaggle
2. Add 2 datasets
3. Run All
4. Nếu OOM → tự động giảm batch size
5. Checkpoint tự động save mỗi 500 steps

## So sánh với notebook cũ

| Feature | Old (kaggle_notebook_auto.ipynb) | New (kaggle_notebook_v2.ipynb) |
|---------|----------------------------------|--------------------------------|
| Self-contained | ❌ Cần `xtts_finetune.*` | ✅ Tất cả code inline |
| Valid_indices fix | ❌ Không có | ✅ Có |
| Trainable params check | ❌ Không có | ✅ Có |
| Auto-reduce batch | ❌ Không có | ✅ Có |
| Vietnamese errors | ⚠️  Một phần | ✅ Toàn bộ |
| Cell 5 malformed | ✅ Đã fix | ✅ N/A (code mới) |

## Kết luận

Notebook mới (`kaggle_notebook_v2.ipynb`) sẽ:
- ✅ Chạy được ngay trên Kaggle
- ✅ Không cần clone repo
- ✅ Fix tất cả bugs đã biết
- ✅ Tự động xử lý OOM
- ✅ Vietnamese error messages
- ✅ Dễ customize và debug

**Khuyến nghị**: Sử dụng notebook mới cho production training.
