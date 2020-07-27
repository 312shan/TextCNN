# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModule import BasicModule
import pickle


class TextCNN(BasicModule):

    def __init__(self, cfg):
        super(TextCNN, self).__init__()
        self.cfg = cfg
        self.embeds = nn.Embedding(cfg.word_num, cfg.word_embedding_dimension)
        if cfg.use_pretrain_embed_weight:
            emb_weights = pickle.load(open(cfg.init_embed_path, 'rb'))
            self.embeds.weight = nn.Parameter(torch.FloatTensor(emb_weights))
        self.drop_emb = nn.Dropout(p=cfg.drop_rate)
        self.conv3 = nn.Conv2d(1, cfg.filter_num, (3, cfg.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, cfg.filter_num, (4, cfg.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, cfg.filter_num, (5, cfg.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.cfg.sentence_max_size - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((self.cfg.sentence_max_size - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((self.cfg.sentence_max_size - 5 + 1, 1))
        self.linear1 = nn.Linear(cfg.filter_num * cfg.label_num, cfg.label_num)

    def forward(self, x):
        batch = x.shape[0]
        x = self.embeds(x)  # 可以把 embedding 放到模型外面节省 gpu 空间
        x = self.drop_emb(x)
        x = x.unsqueeze(1)  # [B,1,T,E],channel is 1
        x1 = torch.tanh(self.conv3(x))  # [B,filter_num,T-win_size_3+1,1]
        x2 = torch.tanh(self.conv4(x))  # [B,filter_num,T-win_size_4+1,1]
        x3 = torch.tanh(self.conv5(x))  # [B,filter_num,T-win_size_5+1,1]

        # Pooling
        x1 = self.Max3_pool(x1)  # [B,filter_num,23,1]=>[B,filter_num,1,1]
        x2 = self.Max4_pool(x2)  # [B, filter_num, 22, 1]=>[B,filter_num,1,1]
        x3 = self.Max5_pool(x3)  # [B, filter_num, 21, 1]=>[B,filter_num,1,1]

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)  # [B, filter_num, 1, 3]
        x = x.view(batch, 1, -1)  # [B,1,filter_num*3]

        # project the features to the labels
        x = self.linear1(x)  # [B,1,filter_num*3] = > [B,1,3]
        x = x.view(-1, self.cfg.label_num)  # [B,3]
        x = torch.softmax(x, dim=-1)
        return x


if __name__ == '__main__':
    print('running the TextCNN...')
