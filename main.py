# coding=utf-8
# cer
import tensorflow as tf
from data import *
from model import Model

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 876
slot_size = 120
intent_size = 21
epoch_num = 10


def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model


def train():
    model = get_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    word2index, index2word, tag2index, index2tag, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    index_train = to_index(train_data_ed, word2index, tag2index, intent2index)
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent = model.step(sess, "train", batch)
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10
                print('Average loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average loss: {}".format(epoch, train_loss))


if __name__ == '__main__':
    train()