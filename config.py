# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding_dimension=50, word_num=20000,
                 epoch=2, sentence_max_size=25, cuda=False,
                 label_num=3, learning_rate=0.01, batch_size=1,
                 filter_num=2):
        self.word_embedding_dimension = word_embedding_dimension  # 词向量的维度
        self.word_num = word_num
        self.filter_num = filter_num
        self.epoch = epoch  # 遍历样本次数
        self.sentence_max_size = sentence_max_size  # 句子长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.cuda = cuda
        self.drop_rate = 0.0  # 试验证明 dropout 帮助不大
        self.init_embed_path = 'data/char2vec.pkl'
        self.use_pretrain_embed_weight = False
