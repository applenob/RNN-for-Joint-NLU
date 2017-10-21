# coding=utf-8
# cer
import tensorflow as tf
from data import *
from model import Model
from metrics import *

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 876
slot_size = 121
intent_size = 22
epoch_num = 50


def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model


def train():
    model = get_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(tf.trainable_variables())
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    print("slot2index: ", slot2index)
    print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent, mask, stepwise_cross_entropy, decoder_targets_one_hot = model.step(sess, "train", batch)
            if i == 0:
                index = random.choice(range(len(batch)))
                print("training debug:")
                print("input:", list(zip(*batch))[0][index])
                print("length:", list(zip(*batch))[1][index])
                print("mask:", mask[index])
                print("target:", list(zip(*batch))[2][index])
                print("decoder_targets_one_hot:")
                # for one in decoder_targets_one_hot[index]:
                #     print(" ".join(map(str, one)))
                print("decoder_prediction:", decoder_prediction[index])
                print("stepwise_cross_entropy:", stepwise_cross_entropy[index])
                print("intent:", list(zip(*batch))[3][index])
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10
                print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))
        # 每训一个epoch，测试一次
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            if j == 0:
                index = random.choice(range(len(batch)))
                print("Input Sentence        : ", index_seq2word(batch[index][0], index2word))
                print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot))
                print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot))
                print("Intent Truth          : ", index2intent[batch[index][3]])
                print("Intent Prediction     : ", index2intent[intent[index]])
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            # print("slot_pred_length: ", slot_pred_length)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            # print(np.shape(true_slot), np.shape(decoder_prediction))
            # print(true_slot, decoder_prediction)
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
            print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))


if __name__ == '__main__':
    train()