# RNN-for-Joint-NLU

## 模型介绍

![](https://github.com/applenob/RNN-for-Joint-NLU/raw/master/res/arc.png)

使用tensorflow r1.3 api，Encoder使用`tf.nn.bidirectional_dynamic_rnn`实现，Decoder使用`tf.contrib.seq2seq.CustomHelper`和`tf.contrib.seq2seq.dynamic_decode`实现。

[原作者Bing Liu的Tensorflow实现](https://github.com/HadoopIt/rnn-nlu)

我的实现相对比较简单，用于学习目的。

## 使用

```
python main.py
```

## 细节

博客文章：
- [Tensorflow动态seq2seq使用总结（r1.3）](https://github.com/applenob/RNN-for-Joint-NLU/blob/master/tensorflow_dynamic_seq2seq.md)
