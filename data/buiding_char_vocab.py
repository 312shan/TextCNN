import pandas as pd
import pickle

df = pd.read_csv('./train.tsv', sep='\t')
df_dev = pd.read_csv('./test.tsv', sep='\t')
df = pd.concat([df, df_dev], axis=0)
print(df.shape)
print(df.head())
df.text_a = df.text_a.map(lambda x: "".join(x.split()))
print(df.head())
char_set = set([c for line in df.text_a.values for c in line])
char2num = {c: ind + 2 for ind, c in enumerate(char_set)}
print(len(char2num))
print(char2num)

with open('char2num.pkl', 'wb') as f:
    pickle.dump(char2num, f)
