# OpenSubtitles YouTube Dataset Pipeline

สร้างและแปลงซับไตเติลจาก YouTube เป็น Dataset สำหรับงาน NLP/ML

## ฟีเจอร์หลัก
- ดาวน์โหลดซับไตเติลอัตโนมัติ (รองรับหลายภาษา)
- แปลงไฟล์ `.vtt` เป็น CSV, JSON, Parquet
- สร้าง parallel dataset (en-th) สำหรับงานแปลภาษา
- Clean ข้อความและลบซ้ำอัตโนมัติ
- All-in-one ใช้งานง่ายในไฟล์เดียว

## วิธีใช้งาน
1. **เตรียม Python Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # หรือ .venv\Scripts\activate บน Windows
   pip install yt-dlp pandas pyarrow
   ```

2. **แก้ไขลิงก์วิดีโอใน `App.py`**
   - เพิ่ม/ลบลิงก์ YouTube ที่ต้องการในตัวแปร `video_urls`
   - กำหนดภาษาซับไตเติลใน `sub_langs` เช่น `['en', 'th']`

3. **รันสคริปต์**
   ```bash
   python App.py
   ```

4. **ผลลัพธ์**
   - ไฟล์ซับไตเติล `.vtt` จะอยู่ในโฟลเดอร์ `subtitles/`
   - Dataset ที่สร้างอัตโนมัติ:
     - `dataset.csv` : ข้อมูลซับไตเติลทั้งหมด
     - `dataset.json` : ข้อมูลซับไตเติลทั้งหมด (JSON)
     - `dataset.parquet` : สำหรับใช้งานกับ HuggingFace Datasets
     - `dataset_parallel.csv` : ข้อมูลคู่แปล en-th (ถ้ามี)
     - `dataset_text_only.txt` : ข้อความล้วน
     - `dataset_text_only_dedup.txt` : ข้อความล้วนแบบลบซ้ำ

## ตัวอย่างโค้ดหลัก (App.py)
```python
if __name__ == '__main__':
    video_urls = [
        'https://www.youtube.com/watch?v=xxxx',
        # ...
    ]
    sub_langs = ['en', 'th']
    download_subtitles(video_urls, sub_langs)
    export_all_vtt_to_datasets()
    export_parallel_dataset()
    export_clean_text()
    export_clean_text_dedup()
```

## ข้อควรระวัง
- วิดีโอบางรายการอาจไม่มีซับไตเติลภาษาไทยหรืออังกฤษ
- ควรติดตั้ง `ffmpeg` เพื่อให้ดาวน์โหลดซับไตเติลได้สมบูรณ์
- ไฟล์ dataset จะถูกเขียนทับทุกครั้งที่รัน

## License
MIT