import tensorflow as tf
import utils
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
import pdb
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import attention_wrapper_ep
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op

FLAGS = utils.FLAGS
num_classes = utils.num_classes
PAD = 0
EOS = 0
GO = 0


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        self.decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.decoder_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_length')

        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [64, 128, 256, 512]
        strides = [1, 2]

        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(self.inputs, 'cnn-1_1', [3, 3], 1, filters[0], [1, 1])
                x = self._batch_norm('bn1_1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2, 2], [2, 2])
                print(x.shape)

                x = self._conv2d(x, 'cnn-1_2', [3, 3], filters[0], filters[1], [1, 1])
                x = self._batch_norm('bn1_2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2, 1], [2, 1])

                print(x.shape)

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2_1', [3, 3], filters[1], filters[2], [1, 1])
                x = self._batch_norm('bn2_1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2, 1], [2, 1])
                print(x.shape)

                x = self._conv2d(x, 'cnn-2_2', [3, 3], filters[2], filters[3], [1, 1])
                x = self._batch_norm('bn2_2', x)
                x = self._leaky_relu(x, 0.01)

                x = self._max_pool(x, [2, 1], [2, 1])
                print(x.shape)

                # [batch_size, max_stepsize, num_features]

                frame_num = 50
                # x = tf.reshape(x, [FLAGS.batch_size,-1,512])
                x = tf.transpose(x, [0, 2, 1, 3])
                x = tf.reshape(x, [FLAGS.batch_size, frame_num, -1])
                x.set_shape([FLAGS.batch_size, frame_num, 1024])
                print(x.shape)
            # x = tf.transpose(x, [0, 2, 1])  # batch_size * 64 * 48
            # shp = x.get_shape().as_list()
            # x.set_shape([FLAGS.batch_size, filters[3], shp[1]])
            # x.set_shape([FLAGS.batch_size, filters[3], 48])

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell

            # *************************layer rnn **************************

        for i in range(FLAGS.rnn_layers):
            with tf.variable_scope('lstm_' + str(i)):
                lstm_fw = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
                lstm_bw = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
                output, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x, scope='bi_lstm' + str(i),
                                                                dtype=tf.float32)
                x = tf.concat(output, axis=2)
                print('lstm_' + str(i) + ':  ', x.get_shape())

                # *************************layer lstm_0 **************************
        outputs = tf.reshape(x, [-1, 2 * FLAGS.num_hidden])  # batch*frame_num x 2*num_hidden
        outputs = tf.reshape(outputs, [FLAGS.batch_size, frame_num, -1])

        with tf.variable_scope("decoder"):
            decoder_embedding_matrix = tf.constant(np.identity(num_classes, dtype=np.float32))
            decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding_matrix, self.decoder_inputs)

            decoder_layers = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
            sequence_length = tf.constant(frame_num, dtype=tf.int32, shape=[FLAGS.batch_size])
            attention_mechanism = attention_wrapper_ep.BahdanauAttention(FLAGS.num_hidden, memory=outputs,
                                                                         normalize=True,
                                                                         memory_sequence_length=sequence_length)

            attn_decoder = attention_wrapper_ep.AttentionWrapper(decoder_layers, attention_mechanism,
                                                                 alignment_history=True,
                                                                 edit_prob_num_class=num_classes)
            fc_layer = tf.layers.Dense(num_classes)

            training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,
                                                                self.decoder_length)

            decoder_initial_state = attn_decoder.zero_state(FLAGS.batch_size, tf.float32).clone(
                cell_state=state[1])  # state[1] is backward direction

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_decoder, helper=training_helper,
                initial_state=decoder_initial_state, output_layer=fc_layer)

            logits, final_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(training_decoder)

            aligments_history = tf.transpose(final_state[4].stack(), [1, 0, 2])  # batch x frame x alpha
            self.aligments_history = aligments_history
            # decoder_logits: [B, T, V]
            decoder_logits = logits.rnn_output



            # [B, T]
            '''
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=num_classes, dtype=tf.float32),
                logits=decoder_logits)

            mask = tf.sequence_mask(self.decoder_length,
                                    maxlen=tf.reduce_max(self.decoder_length),
                                    dtype=tf.float32)

            self.loss = tf.reduce_sum(stepwise_cross_entropy * mask) / tf.reduce_sum(mask)
            '''
            RI_prob_history = tf.transpose(final_state[6].stack(), [1, 0, 2])  # batch x frame x alpha
            edit_probability = self._edit_probability(decoder_logits, RI_prob_history)
            #batch_neg_likelihood = -tf.log(edit_probability)
            batch_neg_likelihood = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.zeros(FLAGS.batch_size),
                    logits=edit_probability)
            self.loss = tf.reduce_sum(batch_neg_likelihood) / FLAGS.batch_size

            tf.summary.scalar("loss", self.loss)

            start_tokens = tf.tile([GO], [FLAGS.batch_size])
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                decoder_embedding_matrix, start_tokens, end_token=EOS)

            inference_decoder_initial_state = attn_decoder.zero_state(
                FLAGS.batch_size, tf.float32).clone(
                cell_state=state[1])

            greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_decoder, helper=inference_helper,
                initial_state=inference_decoder_initial_state, output_layer=fc_layer)

            greedy_decoding_result, _, _2 = tf.contrib.seq2seq.dynamic_decode(
                decoder=greedy_decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=frame_num)

            self.dense_decoded = greedy_decoding_result.sample_id

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size[0], filter_size[1], in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize[0], ksize[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding='SAME',
                              name='max_pool')

    def _edit_probability(self, prob_distribution, R_and_I_prob):
        R_prob = R_and_I_prob[:, :, :3]  # 0 -> consumption, 1->deletion, 2->insertion
        I_prob = R_and_I_prob[:, :, 3:]

        mask = tf.sequence_mask(self.decoder_length,
                                maxlen=tf.reduce_max(self.decoder_length),
                                dtype=tf.bool)
        consum_prob = R_prob[:, :, 0]
        delete_prob = R_prob[:, :, 1]
        insert_prob = tf.add(tf.multiply(tf.cast(mask, tf.float32),
                                         R_prob[:, :, 2]),
                             tf.multiply(tf.cast(tf.logical_not(mask), tf.float32), 1.0))

        maxlen_target = tf.shape(self.decoder_targets)[1]

        edit_prob_graph = tensor_array_ops.TensorArray(
            dtype=tf.float32,
            size=maxlen_target + 1,
            dynamic_size=False)
        edit_prob_row = tensor_array_ops.TensorArray(
            dtype=tf.float32,
            size=maxlen_target + 1,
            dynamic_size=False,
            element_shape=[FLAGS.batch_size])

        initial_row = tf.constant(0, dtype=tf.int32)
        initial_column = tf.constant(0, dtype=tf.int32)

        def condition(row, column):
            return tf.logical_or(
                        tf.greater(maxlen_target, row),
                        tf.greater(maxlen_target, column))

        def body(row, column):

            def f_row_is_zero():
                def col_is_zero():
                    edit_prob_row.write( column,
                        tf.constant(
                            1., tf.float32,
                             [FLAGS.batch_size]))
                    return 0
                def col_not_zero():
                    edit_prob_row.write(column,
                        tf.multiply(
                         edit_prob_row.read(column - 1),
                         delete_prob[:, column]))
                    return 0
                tf.cond(column > 0, col_not_zero, col_is_zero)

                return 0

            def f_row_not_zero():
                insert_prob_current_pos = tf.multiply(
                    tf.gather_nd(I_prob[:, column, :],
                     tf.transpose(
                         tf.stack([tf.range(FLAGS.batch_size),
                                   self.decoder_inputs[:, row]]),
                         [1, 0])),
                    insert_prob[:, column])

                mask_eos = tf.logical_or(
                    tf.cast(self.decoder_inputs[:, row], tf.bool),
                    tf.constant(False, dtype=tf.bool))

                delete_prob_current_pos = tf.add(
                    tf.cast(tf.logical_not(mask_eos), tf.float32) * 1.0,
                    delete_prob[column] * tf.cast(mask_eos, tf.float32))

                consum_prob_current_pos = tf.multiply(
                    tf.gather_nd(prob_distribution[:, column, :],
                                 tf.transpose(
                                     tf.stack([tf.range(FLAGS.batch_size),
                                               self.decoder_inputs[:, row]]),
                                     [1, 0])),
                    consum_prob[:, column])

                def col_is_zero():
                    edit_prob_row.write(column,
                        tf.multiply(insert_prob_current_pos,
                        edit_prob_graph.read(row - 1)[:, column]))
                    return 0
                def col_not_zero():
                    edit_prob_row.write(column,
                    tf.add(
                        tf.multiply(delete_prob_current_pos,
                        edit_prob_row.read(column - 1)),
                    tf.add(
                        tf.multiply(insert_prob_current_pos,
                        edit_prob_graph.read(row - 1)[:, column]),
                        tf.multiply(consum_prob_current_pos,
                        edit_prob_graph.read(row - 1)[:, column - 1]))))
                    return 0

                tf.cond(column > 0, col_not_zero, col_is_zero)
                return 0

            test_y = tf.constant(0, dtype=tf.int32)
            test_x = tf.constant(0, dtype=tf.int32)
            tf.cond(tf.equal(test_x, test_y), f_row_not_zero, f_row_is_zero)

            def f_col_max():
                edit_prob_graph.write(row, edit_prob_row.stack())  # TensorArray, element shape [batch, colmun]
                column = tf.constant(-1, dtype=tf.int32)
                return column

            def f_col_():
                return column

            column = tf.cond(column < maxlen_target, f_col_, f_col_max)

            return (row + 1, column + 1)

        _ = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_row, initial_column],
            parallel_iterations=32,
            swap_memory=True)

        # graph shape after loop is : [row, batch_size, column]
        return tf.gather_nd(
                edit_prob_graph.stack()[:, :, -1],
                tf.transpose(
                  tf.stack([self.decoder_length - 1, tf.range(FLAGS.batch_size)]),
                  [1, 0]))




