import pandas as pd
import re

def clean_text(text):
    # ลบ <00:00:xx.xxx> และ <c>...</c>
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', str(text))
    text = re.sub(r'<c>|</c>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

input_path = 'subtitles/dataset_parallel.csv'
output_path = 'subtitles/dataset_parallel_clean.csv'

df = pd.read_csv(input_path)
df['text_original'] = df['text_original'].map(clean_text)
df['text_thai'] = df['text_thai'].map(clean_text)
df.to_csv(output_path, index=False, encoding='utf-8')
print(f'Exported cleaned parallel dataset to {output_path}')
