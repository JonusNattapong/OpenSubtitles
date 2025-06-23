import pandas as pd

input_path = 'subtitles/dataset_parallel_clean.csv'
output_path = 'subtitles/dataset_parallel_clean_dedup.csv'

df = pd.read_csv(input_path)
df = df.drop_duplicates(subset=['video_id', 'text_original', 'text_thai'])
df.to_csv(output_path, index=False, encoding='utf-8')
print(f'Exported deduplicated parallel dataset to {output_path}')
