import pandas as pd

df = pd.read_csv('./train.tsv', sep='\t')
print(df.head())
print(df.iloc[2, 0])
print(df.iloc[2, 1])
