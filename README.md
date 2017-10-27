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
[Epoch 5] Average train loss: 0.25731880404722735
Input Sentence        :  ['show', 'all', 'airlines', 'with', 'flights', 'between', 'denver', 'and', 'dallas', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
Slot Truth            :  ['O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
Slot Prediction       :  ['O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'I-toloc.city_name', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
Intent Truth          :  atis_airline
Intent Prediction     :  atis_airline
slot accuracy: 0.9753086419753086, intent accuracy: 1.0
slot accuracy: 0.91, intent accuracy: 0.9375
slot accuracy: 0.919431279620853, intent accuracy: 1.0
slot accuracy: 0.9298245614035088, intent accuracy: 0.9375
slot accuracy: 0.9534883720930233, intent accuracy: 0.9375
slot accuracy: 0.9580838323353293, intent accuracy: 1.0
slot accuracy: 0.9319371727748691, intent accuracy: 0.9375
slot accuracy: 0.9565217391304348, intent accuracy: 0.875
slot accuracy: 0.9815950920245399, intent accuracy: 1.0
slot accuracy: 0.958974358974359, intent accuracy: 0.9375
slot accuracy: 0.9356435643564357, intent accuracy: 0.9375
slot accuracy: 0.9473684210526315, intent accuracy: 1.0
slot accuracy: 0.9548022598870056, intent accuracy: 1.0
slot accuracy: 0.9788359788359788, intent accuracy: 1.0
slot accuracy: 0.945054945054945, intent accuracy: 0.9375
slot accuracy: 0.9680851063829787, intent accuracy: 1.0
slot accuracy: 0.9244186046511628, intent accuracy: 1.0
slot accuracy: 0.9470588235294117, intent accuracy: 1.0
slot accuracy: 0.9226190476190477, intent accuracy: 0.875
slot accuracy: 0.9705882352941176, intent accuracy: 1.0
slot accuracy: 0.9333333333333333, intent accuracy: 1.0
slot accuracy: 0.943502824858757, intent accuracy: 0.9375
slot accuracy: 0.94, intent accuracy: 1.0
slot accuracy: 0.9653465346534653, intent accuracy: 0.9375
slot accuracy: 0.9550561797752809, intent accuracy: 1.0
slot accuracy: 0.9178743961352657, intent accuracy: 1.0
slot accuracy: 0.9808917197452229, intent accuracy: 1.0
slot accuracy: 0.9243243243243243, intent accuracy: 0.875
slot accuracy: 0.9508196721311475, intent accuracy: 1.0
slot accuracy: 0.9341317365269461, intent accuracy: 1.0
slot accuracy: 0.9534883720930233, intent accuracy: 0.9375
F1 score for epoch 5: 0.9467727674624227
```

## 细节

博客文章：
- [Tensorflow动态seq2seq使用总结（r1.3）](https://github.com/applenob/RNN-for-Joint-NLU/blob/master/tensorflow_dynamic_seq2seq.md)
