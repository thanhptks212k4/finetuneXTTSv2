# 🚀 Hướng Dẫn Sử Dụng Notebook Self-Contained

## 📋 File nào dùng?

### ✅ **KHUYẾN NGHỊ: `kaggle_notebook_v2.ipynb`**

**Notebook hoàn toàn self-contained** - không cần clone repo!

**Ưu điểm:**
- ✅ Tất cả code trong notebook - không phụ thuộc external packages
- ✅ Dễ customize - edit trực tiếp trong Kaggle
- ✅ Dễ debug - nhìn thấy toàn bộ code
- ✅ Valid_indices sync fix - khắc phục NoneType bug
- ✅ Auto-reduce batch size nếu OOM
- ✅ Trainable params validation
- ✅ Vietnamese error messages

**Nhược điểm:**
- ⚠️  File lớn hơn (~1200 lines)
- ⚠️  Code duplicate (nhưng dễ đọc hơn)

---

### 🔄 **Alternative: `kaggle_notebook_auto.ipynb`**

**Notebook sử dụng `xtts_finetune` package** - cần clone repo

**Ưu điểm:**
- ✅ Code ngắn gọn hơn
- ✅ Tổ chức tốt hơn (modular)
- ✅ Dễ maintain cho developers

**Nhược điểm:**
- ❌ Phải clone repo (Cell 3)
- ❌ Phụ thuộc vào `xtts_finetune.*` package
- ❌ Khó customize trực tiếp

---

## 🎯 Hướng Dẫn Sử Dụng `kaggle_notebook_v2.ipynb`

### Bước 1: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Click **"File" → "Import Notebook"**
4. Paste URL: `https://raw.githubusercontent.com/thanhptks212k4/finetuneXTTSv2/main/kaggle_notebook_v2.ipynb`
5. Hoặc upload file `kaggle_notebook_v2.ipynb` trực tiếp

### Bước 2: Cấu Hình Kaggle

#### 2.1. Bật GPU
- Click **"Accelerator"** ở sidebar phải
- Chọn **"GPU T4 x1"**

#### 2.2. Bật Internet
- Click **"Internet"** ở sidebar phải
- Toggle **ON**

#### 2.3. Add Datasets

**Dataset 1: Audio files**
- Click **"+ Add Data"** ở sidebar phải
- Search: `tinthnhphm21022004/data-speech-to-text`
- Click **"Add"**

**Dataset 2: Manifests**
- Click **"+ Add Data"**
- Search: `thanhphamtien2102224/weight-phowhisper`
- Click **"Add"**

### Bước 3: Chạy Notebook

#### Option A: Run All (Khuyến nghị)
1. Click **"Run All"** ở top menu
2. Đợi ~2-3 giờ
3. Xong!

#### Option B: Run từng cell
1. Click vào Cell 1
2. Press **Shift + Enter** để chạy
3. Lặp lại cho tất cả cells

### Bước 4: Monitor Progress

Notebook sẽ in ra:
```
================================================================================
📦 CELL 1: Cài đặt thư viện
================================================================================
Installing: torch==2.1.0
...
✅ Tất cả thư viện đã được cài đặt!

================================================================================
⚙️  CELL 2: Cấu hình training
================================================================================
✅ Cấu hình hoàn tất
   Effective batch: 2 x 8 = 16

================================================================================
🔧 CELL 3: Kiểm tra GPU
================================================================================
GPU: Tesla T4
VRAM: 15.8 GB
✅ GPU sẵn sàng!

...

================================================================================
🚀 CELL 8: Training loop
================================================================================
Bắt đầu training: 2000 steps
Batch size: 2, Grad accum: 8
================================================================================
Step 50/2000 | Loss: 2.3456 | LR: 2.00e-05
Step 100/2000 | Loss: 2.1234 | LR: 2.00e-05
...
💾 Saved checkpoint: /kaggle/working/output/checkpoints/checkpoint_step_500.pth
...
✅ Training hoàn tất!
```

### Bước 5: Download Kết Quả

Sau khi training xong:

1. **Checkpoint files** ở `/kaggle/working/output/checkpoints/`
   - `final_model.zip` - Model cuối cùng (đã nén)
   - `checkpoint_step_500.pth`, `checkpoint_step_1000.pth`, etc.

2. **Audio samples** ở `/kaggle/working/output/samples/`
   - `sample_1.wav`, `sample_2.wav`, `sample_3.wav`

3. **Download:**
   - Click vào file trong Kaggle file browser (sidebar trái)
   - Click **"⋮"** (3 dots)
   - Click **"Download"**

---

## 🔧 Customization

### Thay đổi Hyperparameters

Edit **Cell 2**:

```python
# Training
BATCH_SIZE = 2              # Giảm xuống 1 nếu OOM
GRAD_ACCUM_STEPS = 8        # Tăng lên 16 nếu giảm batch size
LEARNING_RATE = 2e-5        # Thử 1e-5 hoặc 5e-5
MAX_STEPS = 2000            # Tăng lên 5000 cho better quality

# Memory
USE_FP16 = True             # Giữ True để tiết kiệm VRAM
FREEZE_ENCODER = True       # Giữ True để tiết kiệm VRAM
```

### Thay đổi Validation Texts

Edit **Cell 2**:

```python
VAL_TEXTS = [
    "Câu của bạn ở đây.",
    "Thêm câu thứ hai.",
    "Và câu thứ ba.",
]
```

### Thay đổi Dataset Paths

Nếu bạn dùng datasets khác, edit **Cell 2**:

```python
AUDIO_BASE = '/kaggle/input/your-audio-dataset'
MANIFEST_BASE = '/kaggle/input/your-manifest-dataset'
TRAIN_MANIFEST_SRC = f'{MANIFEST_BASE}/train.jsonl'
TEST_MANIFEST_SRC = f'{MANIFEST_BASE}/test.jsonl'
```

---

## ⚠️ Troubleshooting

### Lỗi: "No GPU detected"
**Giải pháp:**
- Kiểm tra đã bật GPU T4 chưa (sidebar phải)
- Restart notebook và thử lại

### Lỗi: "CUDA out of memory"
**Giải pháp:**
- Notebook sẽ **tự động giảm batch size** từ 2 xuống 1
- Nếu vẫn OOM, edit Cell 2: `BATCH_SIZE = 1`, `GRAD_ACCUM_STEPS = 16`

### Lỗi: "No valid training samples"
**Giải pháp:**
- Kiểm tra đã add đúng 2 datasets chưa
- Kiểm tra paths trong Cell 2
- Xem output của Cell 5 để debug

### Lỗi: "No trainable parameters"
**Giải pháp:**
- Cell 7 sẽ tự động detect và raise error
- Notebook sẽ tự động chuyển sang train toàn bộ model
- Hoặc set `FREEZE_ENCODER = False` trong Cell 2

### Training quá chậm
**Giải pháp:**
- Giảm `MAX_STEPS` xuống 1000 hoặc 500
- Tăng `BATCH_SIZE` lên 4 (nếu VRAM đủ)
- Giảm `GRAD_ACCUM_STEPS` xuống 4

### Muốn resume từ checkpoint
**Giải pháp:**
- Hiện tại notebook chưa support resume
- Cần thêm code load checkpoint trong Cell 7
- Hoặc dùng `kaggle_notebook_auto.ipynb` (có support resume)

---

## 📊 So Sánh 2 Notebooks

| Feature | v2 (Self-Contained) | auto (Package-based) |
|---------|---------------------|----------------------|
| **Setup** | Không cần clone repo | Cần clone repo (Cell 3) |
| **Dependencies** | Tất cả inline | Import từ `xtts_finetune.*` |
| **File size** | ~1200 lines | ~400 lines |
| **Customization** | Dễ - edit trực tiếp | Khó - phải edit package |
| **Debugging** | Dễ - code visible | Khó - code ở package |
| **Valid_indices fix** | ✅ Có | ✅ Có (trong package) |
| **Auto-reduce batch** | ✅ Có | ❌ Không |
| **Trainable params check** | ✅ Có | ⚠️  Có nhưng ở package |
| **Vietnamese errors** | ✅ Toàn bộ | ⚠️  Một phần |
| **Resume training** | ❌ Chưa có | ✅ Có |
| **Maintenance** | Dễ cho users | Dễ cho developers |

---

## 🎓 Hiểu Notebook Structure

### Cell 1: Install Dependencies
- Cài đặt PyTorch, TTS, và các thư viện cần thiết
- Pinned versions để reproducibility

### Cell 2: Configuration
- Tất cả hyperparameters ở đây
- Dễ dàng customize

### Cell 3: GPU Check + Seed
- Validate GPU available
- Set random seed cho reproducibility

### Cell 4: Download Base Model
- Tải XTTS v2 từ HuggingFace
- Verify files downloaded

### Cell 5: Audio Preprocessing
- **CRITICAL**: Fix audio paths (auto-detect)
- Select reference audio (5-10s, highest SR)
- Validate manifests

### Cell 6: Build Dataset
- PyTorch Dataset class
- DataLoader với collate_fn
- Lazy loading để tiết kiệm RAM

### Cell 7: Load Model + Freeze
- Load XTTS model
- Freeze encoder (tiết kiệm VRAM)
- **CRITICAL**: Validate trainable params > 0
- Extract speaker embedding

### Cell 8: Training Loop
- **CRITICAL**: Valid_indices sync fix
- Auto-reduce batch size on OOM
- Gradient accumulation
- FP16 mixed precision
- Checkpoint saving

### Cell 9: Save Final Checkpoint
- Save final model
- Zip for download

### Cell 10: Inference Test
- Generate audio samples
- Test model quality

### Cell 11: Summary
- Print training stats
- List output files

---

## 💡 Tips & Best Practices

### 1. Start Small
- Đầu tiên chạy với `MAX_STEPS = 500` để test
- Nếu OK, tăng lên 2000 hoặc 5000

### 2. Monitor Loss
- Loss nên giảm dần
- Nếu loss không giảm sau 500 steps → có vấn đề

### 3. Check Samples
- Nghe audio samples trong Cell 10
- Nếu quality tốt → model đang học đúng

### 4. Save Checkpoints
- Checkpoint tự động save mỗi 500 steps
- Download checkpoint tốt nhất (lowest loss)

### 5. Experiment
- Thử các learning rates khác nhau
- Thử freeze/unfreeze encoder
- Thử batch sizes khác nhau

---

## 📚 Resources

- **GitHub Repo**: https://github.com/thanhptks212k4/finetuneXTTSv2
- **XTTS v2 Paper**: https://arxiv.org/abs/2406.04904
- **Coqui TTS**: https://github.com/idiap/coqui-ai-TTS
- **Kaggle Docs**: https://www.kaggle.com/docs

---

## 🆘 Support

Nếu gặp vấn đề:

1. **Check Cell Output**: Đọc error messages (tiếng Việt)
2. **Check README_V2.md**: Có giải thích chi tiết về bugs
3. **GitHub Issues**: Tạo issue với full error log
4. **Kaggle Discussion**: Post trong Kaggle discussion

---

## ✅ Checklist Trước Khi Chạy

- [ ] Đã add 2 datasets
- [ ] Đã bật GPU T4
- [ ] Đã bật Internet
- [ ] Đã đọc Cell 2 (config)
- [ ] Đã customize hyperparameters (nếu cần)
- [ ] Đã đọc troubleshooting section
- [ ] Sẵn sàng đợi 2-3 giờ

**Chúc bạn training thành công! 🎉**
