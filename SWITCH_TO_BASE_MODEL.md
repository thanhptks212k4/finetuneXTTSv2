# Chuyển sang XTTS v2 Base Model

## Thay đổi

Đã chuyển từ `anhnh2002/vnTTS` (Vietnamese fine-tuned) sang **`coqui/XTTS-v2`** (official base model).

## Lý do

- **Base model gốc**: Đảm bảo tương thích 100% với Coqui TTS APIs
- **Multilingual**: Hỗ trợ nhiều ngôn ngữ, tokenizer ổn định hơn
- **Official support**: Model chính thức từ Coqui, được maintain tốt
- **Fresh start**: Train từ đầu với data tiếng Việt của bạn

## Files đã update

1. **`xtts_finetune/config.py`**
   ```python
   hf_repo_id: str = "coqui/XTTS-v2"  # Thay vì "anhnh2002/vnTTS"
   ```

2. **`README.md`**
   - Updated base model reference
   - Updated license information

3. **`kaggle_notebook.ipynb`**
   - Cell 6: Updated config to use `coqui/XTTS-v2`

4. **`xtts_finetune/trainer.py`**
   - Fixed tokenizer language error
   - Auto-detect language hoặc dùng first supported language
   - Không còn hardcode `lang="vi"` nữa

## Cách chạy trên Kaggle

### Bước 1: Pull code mới
Trong Kaggle notebook, chạy Cell 2:
```python
!cd /kaggle/working/finetuneXTTSv2 && git pull
```

Hoặc xóa repo cũ và clone lại:
```python
import shutil, os
if os.path.exists('/kaggle/working/finetuneXTTSv2'):
    shutil.rmtree('/kaggle/working/finetuneXTTSv2')
!git clone https://github.com/thanhptks212k4/finetuneXTTSv2.git /kaggle/working/finetuneXTTSv2
```

### Bước 2: Chạy training như bình thường
- Cell 0: Pin numpy (nếu chưa chạy)
- Cell 1: Check GPU
- Cell 2: Clone/update code
- Cell 3-9: Setup
- Cell 10: Training

## Sự khác biệt

### Model cũ (`anhnh2002/vnTTS`)
- ✅ Đã fine-tune cho tiếng Việt
- ❌ Tokenizer không hỗ trợ language code "vi"
- ❌ Có thể có custom modifications không tương thích

### Model mới (`coqui/XTTS-v2`)
- ✅ Official base model, stable APIs
- ✅ Tokenizer hỗ trợ auto-detect language
- ✅ Multilingual: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
- ✅ Bạn sẽ train từ đầu với data tiếng Việt → model hoàn toàn của bạn
- ⚠️ Cần train lâu hơn vì bắt đầu từ base model

## Kỳ vọng

### Lần đầu train
- Model sẽ download từ HuggingFace (~2GB)
- Training time: tương tự như trước
- Quality: Có thể cần nhiều epochs hơn để đạt chất lượng tốt

### Sau khi train xong
- Model sẽ học được đặc trưng tiếng Việt từ data của bạn
- Chất lượng phụ thuộc vào:
  - Số lượng data (càng nhiều càng tốt, tối thiểu 10k samples)
  - Chất lượng audio (clean, 22050 Hz)
  - Số epochs (có thể cần 2-3 epochs thay vì 1)

## Troubleshooting

### Nếu gặp lỗi tokenizer
Code đã được fix để:
1. Thử tokenize không có language code (auto-detect)
2. Nếu fail, thử với first supported language
3. Log chi tiết lỗi để debug

### Nếu muốn quay lại model cũ
Edit `xtts_finetune/config.py`:
```python
hf_repo_id: str = "anhnh2002/vnTTS"
```

Rồi commit và push:
```bash
git add xtts_finetune/config.py
git commit -m "Revert to anhnh2002/vnTTS"
git push
```

## Lợi ích của việc train từ base model

1. **Full control**: Model hoàn toàn của bạn, không phụ thuộc vào fine-tuning của người khác
2. **Customization**: Có thể điều chỉnh mọi aspect của training
3. **Data quality**: Model học trực tiếp từ data của bạn, không bị ảnh hưởng bởi data cũ
4. **Reproducibility**: Dễ reproduce và debug hơn
5. **Licensing**: Rõ ràng về license (Coqui Public Model License)

## Next Steps

1. ✅ Pull code mới từ GitHub
2. ✅ Chạy Cell 2 trên Kaggle để update
3. ✅ Chạy training như bình thường
4. 📊 Monitor training loss - có thể cần train lâu hơn
5. 🎧 Test inference sau mỗi checkpoint
6. 🔧 Điều chỉnh hyperparameters nếu cần:
   - Tăng `epochs_per_patch` lên 2-3
   - Giảm `learning_rate` xuống 1e-5 nếu loss không giảm
   - Tăng `patch_size` nếu có đủ VRAM

---

**Commit**: `1a0bfe8` - "Switch to official coqui/XTTS-v2 base model instead of anhnh2002/vnTTS"

**Files Changed**: 
- `xtts_finetune/config.py`
- `README.md`
- `kaggle_notebook.ipynb`
