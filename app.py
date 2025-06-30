import yt_dlp
import os
import glob
import re
import csv
import pandas as pd
import json
import time
import argparse

def download_subtitles(video_urls, sub_langs, output_dir='subtitles'):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': sub_langs,
        'skip_download': True,
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
    }
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            print(f"Downloading subtitles for {url}")
            try:
                info = ydl.extract_info(url, download=False)
                vid = info.get('id')
                found = False
                for lang in sub_langs:
                    vtt_path = os.path.join(output_dir, f"{vid}.{lang}.vtt")
                    if os.path.exists(vtt_path):
                        print(f"  - Subtitle already exists: {vtt_path}")
                        found = True
                        continue  # ข้ามการดาวน์โหลดถ้ามีไฟล์อยู่แล้ว
                    elif (lang in info.get('subtitles', {}) or lang in info.get('automatic_captions', {})):
                        print(f"  - Downloading subtitle: {lang}")
                        ydl.download([url])
                        found = True
                    else:
                        print(f"  - No subtitle found for language: {lang}")
                results.append({'video_id': vid, 'url': url, 'subtitle_found': found, 'error': None})
            except Exception as e:
                print(f"  - Error: {e}")
                results.append({'video_id': None, 'url': url, 'subtitle_found': False, 'error': str(e)})
            time.sleep(5)  # ดีเลย์ 5 วินาทีระหว่างแต่ละ request
    return results

def parse_vtt(vtt_path):
    entries = []
    with open(vtt_path, encoding='utf-8') as infile:
        start, end, text = None, None, ''
        for line in infile:
            line = line.strip()
            if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} -->', line):
                if start and text:
                    entries.append({'start': start, 'end': end, 'text': text.strip()})
                times = re.findall(r'(\d{2}:\d{2}:\d{2}\.\d{3})', line)
                start, end = times[0], times[1]
                text = ''
            elif line and not line.startswith('WEBVTT') and not line.startswith('Kind:') and not line.startswith('Language:'):
                text += line + ' '
        if start and text:
            entries.append({'start': start, 'end': end, 'text': text.strip()})
    return entries

def export_all_vtt_to_datasets(output_dir='subtitles'):
    vtt_files = glob.glob(f'{output_dir}/*.vtt')
    all_entries = []
    for vtt in vtt_files:
        entries = parse_vtt(vtt)
        for e in entries:
            e['video_id'] = os.path.splitext(os.path.basename(vtt))[0].split('.')[0]
        all_entries.extend(entries)
    # CSV
    csv_path = f'{output_dir}/dataset.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'start', 'end', 'text'])
        writer.writeheader()
        writer.writerows(all_entries)
    # JSON
    json_path = f'{output_dir}/dataset.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    # Parquet
    df = pd.DataFrame(all_entries)
    df.to_parquet(f'{output_dir}/dataset.parquet', index=False)
    print('Exported CSV, JSON, and Parquet datasets in', output_dir)

def align_subs(subs1, subs2):
    d2 = {s['start']: s for s in subs2}
    aligned = []
    for s1 in subs1:
        s2 = d2.get(s1['start'])
        if s2:
            aligned.append((s1['start'], s1['text'], s2['text']))
    return aligned

def export_parallel_dataset(output_dir='subtitles', lang1='en', lang2='th'):
    output_path = f'{output_dir}/dataset_parallel.csv'
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'text_original', 'text_thai'])
        for vtt2 in glob.glob(f'{output_dir}/*.{lang2}.vtt'):
            vid = os.path.basename(vtt2).split('.')[0]
            vtt1 = vtt2.replace(f'.{lang2}.vtt', f'.{lang1}.vtt')
            if os.path.exists(vtt1):
                subs2 = parse_vtt(vtt2)
                subs1 = parse_vtt(vtt1)
                for _, en, th in align_subs(subs1, subs2):
                    writer.writerow([vid, en, th])
    print('Exported parallel dataset to', output_path)

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def export_clean_text(output_dir='subtitles'):
    df = pd.read_csv(f'{output_dir}/dataset.csv')
    cleaned = df['text'].map(clean_text)
    cleaned.to_csv(f'{output_dir}/dataset_text_only.txt', index=False, header=False, encoding='utf-8')
    print('Exported cleaned text dataset to', f'{output_dir}/dataset_text_only.txt')

def export_clean_text_dedup(output_dir='subtitles'):
    with open(f'{output_dir}/dataset_text_only.txt', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    unique_lines = []
    prev = None
    for line in lines:
        if line != prev:
            unique_lines.append(line)
        prev = line
    with open(f'{output_dir}/dataset_text_only_dedup.txt', 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    print('Exported deduplicated text dataset to', f'{output_dir}/dataset_text_only_dedup.txt')

def export_parallel_both_directions(input_path='subtitles/dataset_parallel_long.csv', output_path='subtitles/dataset_parallel_both_directions.csv'):
    """
    รวมชุดข้อมูลแปล EN→TH และ TH→EN เป็น training set เดียว (ลบแถวซ้ำ)
    """
    import pandas as pd
    df = pd.read_csv(input_path)
    # เตรียม EN→TH
    df1 = df[['text_original', 'text_thai']].copy()
    df1.columns = ['src', 'tgt']
    # เตรียม TH→EN
    df2 = df[['text_thai', 'text_original']].copy()
    df2.columns = ['src', 'tgt']
    # รวมและลบซ้ำ
    df_both = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    df_both.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Exported parallel both-directions dataset to {output_path}')

def main():
    parser = argparse.ArgumentParser(description='OpenSubtitles YouTube Dataset Pipeline')
    parser.add_argument('task', nargs='?', default='all', choices=['all', 'download', 'export', 'parallel', 'clean', 'dedup', 'export-both'], help='Task to run')
    args = parser.parse_args()

    video_urls = [
        "https://www.youtube.com/watch?v=tEkYbEkl0No",
        "https://www.youtube.com/watch?v=OsTFVOMNG00",
        "https://www.youtube.com/watch?v=KTwZTYjKR7k",
        "https://www.youtube.com/watch?v=VyvTv42v9As",
        "https://www.youtube.com/watch?v=EdlXcVu1CTs",
        "https://www.youtube.com/watch?v=7WN73qK1E9I",
        "https://www.youtube.com/watch?v=ldizQkuWpDE",
        "https://www.youtube.com/watch?v=N2bwTwRpH1E",
        "https://www.youtube.com/watch?v=OcRWF2OQQcA",
        "https://www.youtube.com/watch?v=NAWTNIipq7w",
        "https://www.youtube.com/watch?v=vjwmAf5Gkxc",
        "https://www.youtube.com/watch?v=0k9fWsiIK14",
        "https://www.youtube.com/watch?v=zsuOSDb7gzQ",
        "https://www.youtube.com/watch?v=XOc64EwpCKs",
        "https://www.youtube.com/watch?v=yhx_VdncWTg",
        "https://www.youtube.com/watch?v=2QSiTkLpSKg",
        "https://www.youtube.com/watch?v=79CCBm7CU3E",
        "https://www.youtube.com/watch?v=cYvi_5qWxpQ",
        "https://www.youtube.com/watch?v=1qXuBrdUj2g",
        "https://www.youtube.com/watch?v=edl2BWLToaI",
        "https://www.youtube.com/watch?v=b0hPOwmSoMM",
        "https://www.youtube.com/watch?v=CnDGLlZEVH4",
        "https://www.youtube.com/watch?v=yEbvxESgutg",
        "https://www.youtube.com/watch?v=atejm2w2jWY",
        "https://www.youtube.com/watch?v=R_E4nEVlyps",
        "https://www.youtube.com/watch?v=ySKHFs2w4ZA",
        "https://www.youtube.com/watch?v=fMomIZZTtys",
        "https://www.youtube.com/watch?v=Er7nORDp-cg",
        "https://www.youtube.com/watch?v=j4z25zj6bmE",
        "https://www.youtube.com/watch?v=LgsJ3V9pIG0",
        "https://www.youtube.com/watch?v=X6fuWIUI_DQ",
        "https://www.youtube.com/watch?v=Xaqnv5EEdYY",
        "https://www.youtube.com/watch?v=mKuxWYla-aE",
        "https://www.youtube.com/watch?v=lrXMfPaQVl0",
        "https://www.youtube.com/watch?v=OIdAx8APMb0",
        "https://www.youtube.com/watch?v=dnn1dMXRmHI",
        "https://www.youtube.com/watch?v=VqEIqpWaJrM",
        "https://www.youtube.com/watch?v=ZuiIvevLg40",
        "https://www.youtube.com/watch?v=ipXmDIJc8Ms",
        "https://www.youtube.com/watch?v=EJDCvjWZgYY",
        "https://www.youtube.com/watch?v=QCaFWrT0j-g",
        "https://www.youtube.com/watch?v=Ps_VcXzrSy8",
        "https://www.youtube.com/watch?v=2TXJrfWNlmM",
        "https://www.youtube.com/watch?v=nABSbtZCCQw",
        "https://www.youtube.com/watch?v=k2ExWxcV0t0",
        "https://www.youtube.com/watch?v=NSsx6mkiaf8",
        "https://www.youtube.com/watch?v=cySmR-P3msM",
        "https://www.youtube.com/watch?v=bZZnSsuRVdU",
        "https://www.youtube.com/watch?v=1COrMVHVE9I",
        "https://www.youtube.com/watch?v=y-hcVMuMwug",
        "https://www.youtube.com/watch?v=8bQujOIHdtI",
        "https://www.youtube.com/watch?v=JMYQmGfTltY",
        "https://www.youtube.com/watch?v=k0r8-5NASVU",
        "https://www.youtube.com/watch?v=D67eWcX2XYQ",
        "https://www.youtube.com/watch?v=pyXcENDTyl0",
        "https://www.youtube.com/watch?v=qJ7Og6w1T0I",
        "https://www.youtube.com/watch?v=vKGOZ3eVSmY",
        "https://www.youtube.com/watch?v=W81_WewjVgc",
        "https://www.youtube.com/watch?v=GVTHxfBUCv4",
        "https://www.youtube.com/watch?v=KUD3a6KKIzA",
        "https://www.youtube.com/watch?v=DdO5rYHIsO4",
        "https://www.youtube.com/watch?v=R7xnGg-RXa0",
        "https://www.youtube.com/watch?v=ztBuHH5m3Zk",
        "https://www.youtube.com/watch?v=tXp_eT_-1EI",
        "https://www.youtube.com/watch?v=1RHjPRpJ7Xs",
        "https://www.youtube.com/watch?v=aY7TUZPCXUY",
        "https://www.youtube.com/watch?v=Zo5kzYO8cGA",
        "https://www.youtube.com/watch?v=wQnuTelS2ig",
        "https://www.youtube.com/watch?v=HKVaI7fBsL4",
        "https://www.youtube.com/watch?v=zVr8RkqJF2Y",
        "https://www.youtube.com/watch?v=O0fMxbRdQbk",
        "https://www.youtube.com/watch?v=_aau9u1snN8",
        "https://www.youtube.com/watch?v=OHCxCYAShtc",
        "https://www.youtube.com/watch?v=2qVB2pljIeo",
        "https://www.youtube.com/watch?v=58NLdFIruM8",
        "https://www.youtube.com/watch?v=HJE2SPTyldE",
        "https://www.youtube.com/watch?v=wz2cYMDPK3k",
        "https://www.youtube.com/watch?v=2xcv7jgtIfY",
        "https://www.youtube.com/watch?v=ykJRXT-Xbhc",
        "https://www.youtube.com/watch?v=Nd1gQ1Qrz6c",
        "https://www.youtube.com/watch?v=26c_st6bC-s",
        "https://www.youtube.com/watch?v=ULdtF87q9cE",
        "https://www.youtube.com/watch?v=LTe57unDbAU",
        "https://www.youtube.com/watch?v=4dxbq7HgWog",
        "https://www.youtube.com/watch?v=kkxUQ-k4HVs",
        "https://www.youtube.com/watch?v=i5A3sSpXXLA",
        "https://www.youtube.com/watch?v=oaD6F83HmBE",
        "https://www.youtube.com/watch?v=9oEFdzgmGxE",
        "https://www.youtube.com/watch?v=b8Ki1EO_5Qc",
        "https://www.youtube.com/watch?v=Cm4oRRyu41I",
        "https://www.youtube.com/watch?v=u391_4lDJO0",
        "https://www.youtube.com/watch?v=MLFm-dsmKa4",
        "https://www.youtube.com/watch?v=FbHDo91XSkY",
        "https://www.youtube.com/watch?v=swiMbbw2odE",
        "https://www.youtube.com/watch?v=epAVydG6IxI",
        "https://www.youtube.com/watch?v=tdQ3f7NpSUY",
        "https://www.youtube.com/watch?v=zltR8heo2y8",
        "https://www.youtube.com/watch?v=oB6F0oW_MjE",
        "https://www.youtube.com/watch?v=4S6T_V92Z8Q",
        "https://www.youtube.com/watch?v=BiQtjkbraus",
        "https://www.youtube.com/watch?v=0iQWOSzRVpI",
        "https://www.youtube.com/watch?v=3SDjwL0-CtI",
        "https://www.youtube.com/watch?v=URtvxQ0gdUU",
        "https://www.youtube.com/watch?v=BmCQ_BowLVQ",
        "https://www.youtube.com/watch?v=5RqFJaeCp_0",
        "https://www.youtube.com/watch?v=OsAlLgGf9JM",
        "https://www.youtube.com/watch?v=2G-6jj5k-KY",
        "https://www.youtube.com/watch?v=CNdIIa8-vKk",
        "https://www.youtube.com/watch?v=OGFB8zAakJU",
        "https://www.youtube.com/watch?v=ZxXruY7llcc",
        "https://www.youtube.com/watch?v=zZ9kseYLAQs",
        "https://www.youtube.com/watch?v=MQ0rsxtiG2Y",
        "https://www.youtube.com/watch?v=CB64ABXGHNU",
        "https://www.youtube.com/watch?v=0GQozcTPyO0",
        "https://www.youtube.com/watch?v=ZUhO0XiHKm4",
        "https://www.youtube.com/watch?v=9-bpWoliv3c",
        "https://www.youtube.com/watch?v=giT0ytynSqg",
        "https://www.youtube.com/watch?v=Pmt6dyH9ZM8",
        "https://www.youtube.com/watch?v=nRvcaWW_IFs",
        "https://www.youtube.com/watch?v=CMzEnLkS2rQ",
        "https://www.youtube.com/watch?v=vm6Io8xFOGM",
        "https://www.youtube.com/watch?v=7IN5soXibEk",
        "https://www.youtube.com/watch?v=Tv1RF9Yfx5g",
        "https://www.youtube.com/watch?v=BlD6rlXXm1k",
        "https://www.youtube.com/watch?v=eZsJ4CpE9Us",
        "https://www.youtube.com/watch?v=0XoMntl4KXI",
        "https://www.youtube.com/watch?v=nzyBaCybPEc",
        "https://www.youtube.com/watch?v=2QaEt_O1-zM",
        "https://www.youtube.com/watch?v=ZHuZ_8VYCWA",
        "https://www.youtube.com/watch?v=rCtvAvZtJyE",
        "https://www.youtube.com/watch?v=Dehoy105Cqc",
        "https://www.youtube.com/watch?v=pZWGr9Jo8ZM",
        "https://www.youtube.com/watch?v=JYLt4gL0GJQ",
        "https://www.youtube.com/watch?v=RK4GYO6XTpc",
        "https://www.youtube.com/watch?v=dabjSeizQOg",
        "https://www.youtube.com/watch?v=tK7nkNCGVb4",
        "https://www.youtube.com/watch?v=rBM6lGk4-fk",
        "https://www.youtube.com/watch?v=oIiv_335yus",
        "https://www.youtube.com/watch?v=HAJ-XLO724A",
        "https://www.youtube.com/watch?v=C3376VDBj5g",
        "https://www.youtube.com/watch?v=zECoaEZRRFU",
        "https://www.youtube.com/watch?v=oP7iC832uO8",
        "https://www.youtube.com/watch?v=E0LcGGHI8dg",
        "https://www.youtube.com/watch?v=G9xKF4HSASY",
        "https://www.youtube.com/watch?v=xW8bEXGmo6o",
        "https://www.youtube.com/watch?v=jRPpwomXHl4",
        "https://www.youtube.com/watch?v=53vyeXlfalw",
        "https://www.youtube.com/watch?v=8c6fU4r_7AI",
        "https://www.youtube.com/watch?v=40jhromlYmI",
        "https://www.youtube.com/watch?v=4eC2na-JTCg",
        "https://www.youtube.com/watch?v=7mUuQezlHhA",
        "https://www.youtube.com/watch?v=clW_td9wyWc",
        "https://www.youtube.com/watch?v=T38lVbnNlkE",
        "https://www.youtube.com/watch?v=l5LbnGOcH4g",
        "https://www.youtube.com/watch?v=ZznpMh0DegE",
        "https://www.youtube.com/watch?v=RA0P7POHgXU",
        "https://www.youtube.com/watch?v=LUjn3RpkcKY",
        "https://www.youtube.com/watch?v=hk79n61Wj9A",
        "https://www.youtube.com/watch?v=k7LmBUqauFc",
        "https://www.youtube.com/watch?v=fk4yEilYu_0",
        "https://www.youtube.com/watch?v=BWC1tSWFcHo",
        "https://www.youtube.com/watch?v=kgU8hXMSebg",
        "https://www.youtube.com/watch?v=0G9EV1I5ICI",
        "https://www.youtube.com/watch?v=lpccPi4DcLg",
        "https://www.youtube.com/watch?v=p_Yg8yXUhLQ",
        "https://www.youtube.com/watch?v=7yP_1IPHycM",
        "https://www.youtube.com/watch?v=aXm9mVKw3OQ",
        "https://www.youtube.com/watch?v=gIpCy7GOdfs",
        "https://www.youtube.com/watch?v=zzUDW5AANRc",
        "https://www.youtube.com/watch?v=ENBLyarDmU8",
        "https://www.youtube.com/watch?v=XVhu9JdyIu0",
        "https://www.youtube.com/watch?v=Z1SxgWq3Gso",
        "https://www.youtube.com/watch?v=Kxo1lEy4xec",
        "https://www.youtube.com/watch?v=fhSKHLJGkE8",
        "https://www.youtube.com/watch?v=4pXC4gK65Fc",
        "https://www.youtube.com/watch?v=UmSj4_6Y4mE",
        "https://www.youtube.com/watch?v=fGGc6PiCggA",
        "https://www.youtube.com/watch?v=atnJEa4e-fU",
        "https://www.youtube.com/watch?v=yHCtfU3syM4",
        "https://www.youtube.com/watch?v=URV-_oEr-kg",
        "https://www.youtube.com/watch?v=0xZLgTq3bps",
        "https://www.youtube.com/watch?v=qPXAfouXa_I",
        "https://www.youtube.com/watch?v=icsGyYkNa2Q",
        "https://www.youtube.com/watch?v=IHuwxzVRamg",
        "https://www.youtube.com/watch?v=mexrsdjfqXM",
        "https://www.youtube.com/watch?v=ASliCpc54ho",
        "https://www.youtube.com/watch?v=qekTIG9jBsA",
        "https://www.youtube.com/watch?v=RFLO38p5HMM",
        "https://www.youtube.com/watch?v=obMCmJcSoqA",
        "https://www.youtube.com/watch?v=uxu37dqVR90",
        "https://www.youtube.com/watch?v=TPUoxA2rnSI",
        "https://www.youtube.com/watch?v=zXLQzoX-bhg",
        "https://www.youtube.com/watch?v=SpqZ-d2_--U",
        "https://www.youtube.com/watch?v=li70iz1NaDY",
        "https://www.youtube.com/watch?v=G4hkYDjPSFs",
        "https://www.youtube.com/watch?v=Fg7U-BhiZGE",
        "https://www.youtube.com/watch?v=8vkPRm-TueY",
        "https://www.youtube.com/watch?v=M4Yb6a7FCmg",
        "https://www.youtube.com/watch?v=lCb6ZfHCwJw",
        "https://www.youtube.com/watch?v=yUe3AyLCGUs",
        "https://www.youtube.com/watch?v=0tEqrZ0eRd0",
        "https://www.youtube.com/watch?v=ip4zspw-bwI",
        "https://www.youtube.com/watch?v=Y2IoFq1f-Yk",
        "https://www.youtube.com/watch?v=EUfky0jzRU8",
        "https://www.youtube.com/watch?v=WRfSTTgVPf8",
        "https://www.youtube.com/watch?v=ub1I0wyRgZ8",
        "https://www.youtube.com/watch?v=Lp1X33qAXls",
        "https://www.youtube.com/watch?v=OCDdq0FpCDM",
        "https://www.youtube.com/watch?v=f7O178kaYVs",
        "https://www.youtube.com/watch?v=FWxCj1W5AWo",
        "https://www.youtube.com/watch?v=yNvyJPqWGys",
        "https://www.youtube.com/watch?v=XD4SnBDZBIE",
        "https://www.youtube.com/watch?v=ndZDQlqvG7I",
        "https://www.youtube.com/watch?v=mlrCz3fF6js",
        "https://www.youtube.com/watch?v=3gnrkZJzs-w",
        "https://www.youtube.com/watch?v=W4tqbEmplug",
        "https://www.youtube.com/watch?v=4yohVh4qcas",
        "https://www.youtube.com/watch?v=jA3Pa9E2n-g",
    ]
    sub_langs = ['en', 'th']

    if args.task == 'download' or args.task == 'all':
        print('=== Downloading subtitles ===')
        download_subtitles(video_urls, sub_langs)
    if args.task == 'export' or args.task == 'all':
        print('=== Exporting all VTT to datasets ===')
        export_all_vtt_to_datasets()
    if args.task == 'parallel' or args.task == 'all':
        print('=== Exporting parallel dataset (en-th) ===')
        export_parallel_dataset()
    if args.task == 'clean' or args.task == 'all':
        print('=== Exporting cleaned text dataset ===')
        export_clean_text()
    if args.task == 'dedup' or args.task == 'all':
        print('=== Exporting deduplicated text dataset ===')
        export_clean_text_dedup()
    if args.task == 'export-both' or args.task == 'all':
        print('=== Exporting parallel both-directions dataset ===')
        export_parallel_both_directions()

if __name__ == '__main__':
    main()
