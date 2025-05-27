import numpy as np
import pandas as pd

df_1 = pd.read_csv('validation.csv')
df_2 = pd.read_csv('train.csv')
df_3 = pd.read_csv('test.csv')

df = pd.concat([df_1, df_2, df_3])
df = df.dropna()
df = df.reset_index(drop=True)

if 'comment' in df.columns:
    df = df.rename(columns={'comment': 'text'})

print(df.head())
print('\nPhân bố nhãn')
print(df['label'].value_counts())

df[['text', 'label']].to_csv('merged_data.csv', index=False)