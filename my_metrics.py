# coding=utf-8
# @author: cer
import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def accuracy_score(true_data, pred_data, true_length=None):
    true_data = np.array(true_data)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape
    if true_length is not None:
        val_num = np.sum(true_length)
        assert val_num != 0
        res = 0
        for i in range(true_data.shape[0]):
            res += np.sum(true_data[i, :true_length[i]] == pred_data[i, :true_length[i]])
    else:
        val_num = np.prod(true_data.shape)
        assert val_num != 0
        res = np.sum(true_data == pred_data)
    res /= float(val_num)
    return res


def get_data_from_sequence_batch(true_batch, pred_batch, padding_token):
    """从序列的batch中提取数据：
    [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]"""
    true_ma = ma.masked_equal(true_batch, padding_token)
    pred_ma = ma.masked_array(pred_batch, true_ma.mask)
    true_ma = true_ma.flatten()
    pred_ma = pred_ma.flatten()
    true_ma = true_ma[~true_ma.mask]
    pred_ma = pred_ma[~pred_ma.mask]
    return true_ma, pred_ma


def f1_for_sequence_batch(true_batch, pred_batch, average="micro", padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    labels = list(set(true))
    return f1_score(true, pred, labels=labels, average=average)


def accuracy_for_sequence_batch(true_batch, pred_batch, padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    return accuracy_score(true, pred)