# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import sys


class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        # 每句输入的实际长度，除了padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='decoder_targets')
        self.intent_targets = tf.placeholder(tf.int32, [self.batch_size],
                                             name='intent_targets')

    def build(self):

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],
                                                        -0.1, 0.1), dtype=tf.float32, name="embedding")

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        # Encoder

        # 使用单个LSTM cell
        encoder_f_cell = LSTMCell(self.hidden_size)
        encoder_b_cell = LSTMCell(self.hidden_size)
        # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
        # 下面四个变量的尺寸：B*T*D，B*T*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        # Decoder
        decoder_cell = LSTMCell(self.hidden_size * 2)
        decoder_lengths = self.encoder_inputs_actual_length
        self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
        intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")

        # 求intent
        intent_logits = tf.add(tf.matmul(encoder_final_state_h, intent_W), intent_b)
        # intent_prob = tf.nn.softmax(intent_logits)
        self.intent = tf.argmax(intent_logits, axis=1)

        sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='SOS') * 2
        sos_step_embedded = tf.nn.embedding_lookup(self.embeddings, sos_time_slice)
        # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        pad_step_embedded = tf.zeros([self.batch_size, self.hidden_size*2+self.embedding_size])

        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            # initial_input = sos_step_embedded
            print("initial_input: ", initial_input)
            # 将上面encoder的最终state传入decoder
            initial_cell_state = self.encoder_final_state
            initial_cell_output = None
            initial_loop_state = None
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, self.slot_W), self.slot_b)
                print("slot output_logits: ", output_logits)
                prediction = tf.argmax(output_logits, axis=1)
                pred_embedding = tf.nn.embedding_lookup(self.embeddings, prediction)
                next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
                print("next_input: ", next_input)
                return next_input

            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            # defining if corresponding sequence has ended

            # 输入是h_i+o_{i-1}
            # input_ = tf.concat((next_input, encoder_outputs[time]), 1)
            # input_ = next_input
            # print("input_: ", input_)
            finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            input_ = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished,
                    input_,
                    state,
                    output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            print("time: ", time)
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output,
                                          previous_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn, scope="raw_rnn")
        decoder_outputs = decoder_outputs_ta.stack()
        print("decoder_outputs", decoder_outputs)
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.slot_W), self.slot_b)
        self.decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps,
                                                          decoder_batch_size, self.slot_size))
        # decoder_prob = tf.nn.softmax(decoder_logits)

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        # loss function

        self.decoder_targets_true_length = self.decoder_targets[:, :decoder_max_steps]
        self.decoder_targets_one_hot = tf.one_hot(self.decoder_targets_true_length,
                                                  depth=self.slot_size, dtype=tf.float32)
        print("decoder_targets_true_length: ", self.decoder_targets_true_length)
        print("self.decoder_targets_one_hot: ", self.decoder_targets_one_hot)

        # stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=self.decoder_targets_one_hot,
        #     logits=self.decoder_logits)
        #
        # # Mask the losses
        self.mask = tf.sign(tf.to_float(self.decoder_targets_true_length))
        # self.stepwise_cross_entropy = tf.transpose(stepwise_cross_entropy, [1, 0])
        # masked_losses = self.mask * self.stepwise_cross_entropy
        # mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(decoder_lengths)
        # loss_slot = tf.reduce_mean(mean_loss_by_example)

        loss_slot = tf.contrib.seq2seq.sequence_loss(
            targets=self.decoder_targets_true_length,
            logits=self.decoder_logits,
            weights=self.mask,
            name="slot_loss"
            )

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
            logits=intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)

        self.loss = loss_slot + loss_intent
        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.grads, self.vars))
        # self.train_op = optimizer.minimize(self.loss)

    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                            self.intent, self.mask, self.decoder_logits, self.slot_W]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1],
                         self.decoder_targets: unziped[2],
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.decoder_prediction, self.intent]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results
