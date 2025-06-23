import pandas as pd

input_path = 'subtitles/dataset_parallel_clean_dedup.csv'
long_path = 'subtitles/dataset_parallel_long.csv'
short_path = 'subtitles/dataset_parallel_short.csv'

# อ่านไฟล์
all_df = pd.read_csv(input_path)

# แยกข้อความตามลำดับบรรทัด: สลับ long/short
long_rows = all_df.iloc[1::2]  # เริ่มที่บรรทัดที่ 2 (index 1), ข้ามทีละ 2
short_rows = all_df.iloc[0::2] # เริ่มที่บรรทัดที่ 1 (index 0), ข้ามทีละ 2

long_rows.to_csv(long_path, index=False, encoding='utf-8')
short_rows.to_csv(short_path, index=False, encoding='utf-8')
print(f'Exported long sentences to {long_path}')
print(f'Exported short sentences to {short_path}')
