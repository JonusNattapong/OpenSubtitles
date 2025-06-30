# OpenSubtitles YouTube Dataset Pipeline

สร้างและแปลงซับไตเติลจาก YouTube เป็น Dataset สำหรับงาน NLP/ML เช่น Machine Translation, Text Mining, หรือ Data Augmentation

---

## ฟีเจอร์หลัก

- ดาวน์โหลดซับไตเติลอัตโนมัติจาก YouTube (รองรับหลายภาษา เช่น en, th)
- แปลงไฟล์ `.vtt` เป็น CSV, JSON, Parquet
- สร้าง parallel dataset (en-th) สำหรับงานแปลภาษา
- Clean ข้อความ (ลบ tag, timestamp, whitespace เกิน, ฯลฯ)
- Dedup ข้อความซ้ำ (ทั้งแบบข้อความเดี่ยวและแบบคู่แปล)
- สร้างชุดข้อมูลแปลสองทาง (Thai→English และ English→Thai) ในไฟล์เดียว
- All-in-one ใช้งานง่ายในไฟล์เดียว หรือเลือกทำแต่ละขั้นตอนผ่าน CLI

---

## วิธีใช้งาน

### 1. เตรียม Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # หรือ .venv\Scripts\activate บน Windows
pip install yt-dlp pandas pyarrow
```

### 2. แก้ไขลิงก์วิดีโอใน `App.py`
- เพิ่ม/ลบลิงก์ YouTube ที่ต้องการในตัวแปร `video_urls`
- กำหนดภาษาซับไตเติลใน `sub_langs` เช่น `['en', 'th']`

### 3. ตัวอย่างการใช้งาน CLI

```bash
python App.py download         # ดาวน์โหลดซับไตเติล
python App.py export           # แปลง VTT เป็น dataset
python App.py parallel         # สร้าง parallel dataset (en-th)
python App.py clean            # Clean ข้อความ
python App.py dedup            # Dedup ข้อความ
python App.py export-both      # สร้างชุดแปลสองทาง (en-th, th-en)
python App.py                  # (หรือ python App.py all) ทำทุกขั้นตอนแบบ all-in-one
```

---

## ผลลัพธ์ที่ได้

- ไฟล์ซับไตเติล `.vtt` จะอยู่ในโฟลเดอร์ `subtitles/`
- Dataset ที่สร้างอัตโนมัติ:
  - `dataset.csv` : ข้อมูลซับไตเติลทั้งหมด
  - `dataset.json` : ข้อมูลซับไตเติลทั้งหมด (JSON)
  - `dataset.parquet` : สำหรับใช้งานกับ HuggingFace Datasets
  - `dataset_parallel.csv` : ข้อมูลคู่แปล en-th (ถ้ามี)
  - `dataset_text_only.txt` : ข้อความล้วน
  - `dataset_text_only_dedup.txt` : ข้อความล้วนแบบลบซ้ำ
  - `dataset_parallel_both_directions.csv` : ข้อมูลคู่แปลสองทาง (Thai→English และ English→Thai)

---

## ข้อมูลชุด parallel (bidirectional)

- `dataset_parallel_both_directions.csv` :
  - รวมทั้งคู่แปล EN→TH และ TH→EN ในไฟล์เดียว (column: src, tgt)
  - เหมาะสำหรับ train โมเดลแปลสองทาง (bidirectional NMT)
  - ถ้าต้องการ dedup เฉพาะ src หรือ tgt ให้แจ้งเพิ่มได้

---

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
    export_parallel_both_directions()
```

---

## หมายเหตุเกี่ยวกับประโยคซ้ำและการ dedup

- ถ้า src, tgt เหมือนกันเป๊ะ จะถูกลบซ้ำโดยอัตโนมัติ
- ถ้า src เหมือนกันแต่ tgt ต่าง (หรือมีข้อความต่อเนื่อง/overlap) จะไม่ถือว่าซ้ำ
- ถ้าต้องการ dedup เฉพาะ src หรือ fuzzy match แจ้งได้

---

## ข้อควรระวัง

- วิดีโอบางรายการอาจไม่มีซับไตเติลภาษาไทยหรืออังกฤษ
- ควรติดตั้ง `ffmpeg` เพื่อให้ดาวน์โหลดซับไตเติลได้สมบูรณ์ (โดยเฉพาะถ้าต้องการแปลงไฟล์วิดีโอ/เสียง)
- ไฟล์ dataset จะถูกเขียนทับทุกครั้งที่รัน
- หากต้องการปรับ logic การ clean/dedup/align สามารถแก้ไขโค้ดใน `App.py` ได้โดยตรง

---

## License

MIT