# coding=utf-8
# cer
import numpy as np


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