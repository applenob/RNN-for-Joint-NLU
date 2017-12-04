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

输出：
```
[Epoch 27] Average train loss: 0.0
Input Sentence        :  ['what', 'are', 'the', 'flights', 'and', 'prices', 'from', 'la', 'to', 'charlotte', 'for', 'monday', 'morning']
Slot Truth            :  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'O', 'B-depart_date.day_name', 'B-depart_time.period_of_day']
Slot Prediction       :  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'O', 'B-depart_date.day_name', 'B-depart_time.period_of_day']
Intent Truth          :  atis_flight
Intent Prediction     :  atis_flight#atis_airfare
Intent accuracy for epoch 27: 0.969758064516129
Slot accuracy for epoch 27: 0.9782146713160718
Slot F1 score for epoch 27: 0.977950943062074
[Epoch 28] Average train loss: 0.0
Input Sentence        :  ['show', 'me', 'the', 'last', 'flight', 'from', 'love', 'field']
Slot Truth            :  ['O', 'O', 'O', 'B-flight_mod', 'O', 'O', 'B-fromloc.airport_name', 'I-fromloc.airport_name']
Slot Prediction       :  ['O', 'O', 'O', 'B-flight_mod', 'O', 'O', 'B-fromloc.airport_name', 'I-fromloc.airport_name']
Intent Truth          :  atis_flight
Intent Prediction     :  atis_flight
Intent accuracy for epoch 28: 0.9717741935483871
Slot accuracy for epoch 28: 0.9794670271393975
Slot F1 score for epoch 28: 0.9792847025495751
```

## 细节

博客文章：
- [Tensorflow动态seq2seq使用总结（r1.3）](https://github.com/applenob/RNN-for-Joint-NLU/blob/master/tensorflow_dynamic_seq2seq.md)
