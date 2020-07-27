# -*- coding: utf-8 -*-

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import Config
from data import *
from model import TextCNN

if __name__ == '__main__':
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0009)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--filter_num', type=int, default=2)
    parser.add_argument('--label_num', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    # Create the configuration
    config = Config(sentence_max_size=25,
                    batch_size=args.batch_size,
                    word_num=11000,
                    label_num=args.label_num,
                    learning_rate=args.lr,
                    cuda=args.gpu,
                    epoch=args.epoch,
                    filter_num=args.filter_num
                    )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    training_set = TextCharDataset(config, path='data/train.tsv', vocab_file='./data/char2num.pkl')
    deving_set = TextCharDataset(config, path='data/dev.tsv', vocab_file='./data/char2num.pkl')

    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=2)
    deving_iter = data.DataLoader(dataset=deving_set,
                                  batch_size=config.batch_size,
                                  num_workers=2)

    config.word_num = len(training_set.tok2num)
    model = TextCNN(config)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    training_lossse = []
    # Train the model
    for epoch in range(config.epoch):
        model.train()
        for data, label in training_iter:

            if config.cuda and torch.cuda.is_available():
                data = data.cuda()
                labels = label.cuda()

            out = model(data)
            loss = criterion(out, label)
            training_lossse.append(loss.item())

            if len(training_lossse) % 100 == 0:
                print("train epoch", epoch, end='  ')
                print("The loss is: %.5f" % (np.average(training_lossse[-100:])))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # run validation on each epoch end
        dev_losses = []
        correct_cnt = []
        len(deving_set)
        with torch.set_grad_enabled(False):
            model.eval()
            for data, label in deving_iter:
                if config.cuda and torch.cuda.is_available():
                    data = data.cuda()
                    labels = label.cuda()

                out = model(data)
                loss = criterion(out, label)
                corrects = (torch.argmax(out, dim=-1) == label).sum().item()
                correct_cnt.append(corrects)
                dev_losses.append(loss.item())
                # save the model in every epoch
            print("The epoch %d dev loss is: %.5f" % (epoch, np.average(dev_losses)))
            print("The epoch %d dev acc is: %.5f" % (epoch, sum(correct_cnt) / (len(dev_losses) * config.batch_size)))
            model.save('checkpoints/epoch{}.ckpt'.format(epoch))
