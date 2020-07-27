import torch
from torch.utils import data
import os
import pickle
import pandas as pd

"""
label	text_a
"""


class TextDataset(data.Dataset):

    def __init__(self, config, path='train.csv', vocab_file='vocab.txt', ):
        self.fname = path
        self.config = config
        self.vocab_file = vocab_file
        self.df = pd.read_csv(self.fname, sep='\t')

        print(self.df.head())
        with open(self.vocab_file, encoding='utf-8') as f:
            self.tok2num = {l[:-1]: ind + 2 for ind, l in enumerate(f)}
        self.tok2num['unk'] = 1
        self.tok2num['mask'] = 0
        self.df['text_a'] = self.df['text_a'].map(lambda x: [self.tok2num.get(w, 1) for w in x.split()])

    def __getitem__(self, index):
        maxl = self.config.sentence_max_size
        tok_id_seqs = self.df.iloc[index, 1]
        tok_id_seqs = tok_id_seqs[:maxl]
        if len(tok_id_seqs) < maxl:
            tok_id_seqs += [0] * (maxl - len(tok_id_seqs))

        tok_id_seqs = torch.LongTensor(tok_id_seqs)
        return tok_id_seqs, self.df.iloc[index, 0]

    def __len__(self):
        return len(self.df) - 1


class TextCharDataset(data.Dataset):

    def __init__(self, config, path='train.csv', vocab_file='char2num.pkl', ):
        self.fname = path
        self.config = config
        self.vocab_file = vocab_file
        self.df = pd.read_csv(self.fname, sep='\t')
        self.df.text_a = self.df.text_a.map(lambda x: ''.join(x.split()))
        print(self.df.head())

        self.tok2num = pickle.load(open(vocab_file, 'rb'))
        self.tok2num['unk'] = 1
        self.tok2num['mask'] = 0
        self.df['text_a'] = self.df['text_a'].map(lambda x: [self.tok2num.get(c, 1) for c in x])

    def __getitem__(self, index):
        maxl = self.config.sentence_max_size
        tok_id_seqs = self.df.iloc[index, 1]
        tok_id_seqs = tok_id_seqs[:maxl]
        if len(tok_id_seqs) < maxl:
            tok_id_seqs += [0] * (maxl - len(tok_id_seqs))

        tok_id_seqs = torch.LongTensor(tok_id_seqs)
        return tok_id_seqs, self.df.iloc[index, 0]

    def __len__(self):
        return len(self.df) - 1
