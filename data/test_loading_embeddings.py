import numpy as np
from tqdm import tqdm
import pickle

# 属于数据集的 char 集合
char2num = pickle.load(open('char2num.pkl', 'rb'))
char2vec = np.random.randn(len(char2num) + 2, 300)
print(char2vec.shape)
# load Glove format Vectors
embeddings_index = {}
EMBEDDING_DIM = 300
targeted_cnt = 0
embfile = 'sgns.weibo.bigram-char'
with open(embfile, encoding='utf-8') as f:
    title = f.readline()
    print(title)
    for i, line in tqdm(enumerate(f)):
        values = line.split()
        words = values[:-EMBEDDING_DIM]
        word = ''.join(words)
        coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
        embeddings_index[word] = coefs
        if len(word) == 1 and word in char2num:
            char2vec[char2num[word]] = coefs
            targeted_cnt += 1
print('Found %s word vectors.' % len(embeddings_index))
print('Found %s in case char vectors.' % len(char2vec))
print('targeted char count {} '.format(targeted_cnt))
pickle.dump(char2vec, open('char2vec.pkl', 'wb'))
