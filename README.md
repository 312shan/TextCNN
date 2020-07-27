## TextCNN 短文本中文情感识别试验总结
1. 短文本情况下字级别比单词级别好
2. 增加 filter_num 有帮助
3. 降低学习率
4. 预训练词向量在这个情况下帮助不大
5. dropout 帮助不大
6. 迄今为止影响最大的部分|激活函数|提升四个点

## 当前效果
### 使用 relu 激活函数 
```python
The epoch 29 dev loss is: 0.63871
The epoch 29 dev acc is: 0.90349
```
### 使用 tanh 激活函数
```python
The epoch 29 dev loss is: 0.60318
The epoch 29 dev acc is: 0.94393
```


## TODO
1. 对比不同损失函数

## 参考
[TextCNN paper](https://arxiv.org/abs/1408.5882)  
[TextCNN 原理解析](https://zhuanlan.zhihu.com/p/77634533)  
[加载预训练词向量](https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222)  
