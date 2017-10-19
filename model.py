# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


class Encoder:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, input_steps],
                                             name='encoder_inputs')
        # 每句输入的实际长度，除了padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='encoder_inputs')

    def forward(self):

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],
                                                        -1.0, 1.0), dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        # 使用单个LSTM cell
        encoder_cell = LSTMCell(self.hidden_size)
        # 下面四个变量的尺寸：B*T*D，B*T*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=False)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )


class Decoder:
    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):

        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

    def forward(self):
        slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -1, 1),
                               dtype=tf.float32, name="intent_W")
        intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")
