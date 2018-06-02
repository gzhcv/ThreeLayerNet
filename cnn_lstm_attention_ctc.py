import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages



FLAGS = utils.FLAGS
num_classes = utils.num_classes


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # image
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
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
                x = self._conv2d(self.inputs, 'cnn-1_1', [3,3], 1, filters[0], [1,1])
                x = self._batch_norm('bn1_1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,2], [2,2])
                print(x.shape)

                x = self._conv2d(x, 'cnn-1_2', [3,3], filters[0], filters[1], [1,1])
                x = self._batch_norm('bn1_2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,1], [2,1])
                
                print(x.shape)
                

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2_1', [3,3], filters[1], filters[2], [1,1])
                x = self._batch_norm('bn2_1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, [2,1], [2,1])
                print(x.shape)

                x = self._conv2d(x, 'cnn-2_2', [3,3], filters[2], filters[3], [1,1])
                x = self._batch_norm('bn2_2', x)
                x = self._leaky_relu(x, 0.01)

                x = self._max_pool(x, [2,1], [2,1])
                print(x.shape)

            # [batch_size, max_stepsize, num_features]
            
                frame_num = FLAGS.max_stepsize
                #x = tf.reshape(x, [FLAGS.batch_size,-1,512])
                x = tf.transpose(x, [0, 2, 1,3])
                x = tf.reshape(x, [FLAGS.batch_size, frame_num, -1])
                x.set_shape([FLAGS.batch_size, frame_num, 1024])
                print(x.shape)
            #x = tf.transpose(x, [0, 2, 1])  # batch_size * 64 * 48
            #shp = x.get_shape().as_list()
            #x.set_shape([FLAGS.batch_size, filters[3], shp[1]])
            #x.set_shape([FLAGS.batch_size, filters[3], 48])

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell

                    #*************************layer rnn **************************

        for i in range(FLAGS.rnn_layers):            
            with tf.variable_scope('lstm_' + str(i)):
                lstm_fw = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
                lstm_bw = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
                output, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, x , scope='bi_lstm' + str(i), dtype=tf.float32)
                x = tf.concat(output, axis=2)
                print('lstm_'+ str(i) +':  ', x.get_shape())
    
                    #*************************layer lstm_0 **************************
            
        outputs = tf.reshape(x, [-1, 2 * FLAGS.num_hidden])   # batch*frame_num x 2*num_hidden

        W_s = tf.Variable(tf.truncated_normal ([2 * FLAGS.num_hidden, num_classes], stddev=0.1), name="W_s") 
        b_s = tf.Variable(tf.constant(0., shape=[num_classes]), name="b_s") 

        # Doing the affine projection 
        y = tf.matmul(outputs, W_s) + b_s
            # Reshaping back to the original shape
        y = tf.reshape(y, [FLAGS.batch_size, -1, num_classes]) # batch x 2*num_hidden* x num_class

        # Calculate alpha coefficient
        W_y = tf.Variable(tf.truncated_normal ([num_classes, 1], stddev=0.1), name="W_y")
        V_y = tf.Variable(tf.truncated_normal ([2 * FLAGS.num_hidden, 1], stddev=0.1), name="W_y")
        b_y = tf.Variable(tf.constant(0., shape=[1]), name="b_y")

        y_0 = tf.Variable(tf.zeros([FLAGS.batch_size, 1, num_classes]))
        y = tf.concat([y, y_0], 1)

        #y_idex_num = tf.range(0, (y.shape[1]-1) * y.shape[0] * y.shape[2])
        #y_idex_flattened = tf.add( tf.mul( tf.floordiv(y_idex_num, y.shape[1]-1 ), y.shape[1]), tf.mod(y_idex_num, y.shape[1]-1) )  
        #y = tf.gather(tf.reshape(y, [-1]), y_idex_flattened)
        y = y[:, 1:, :]

        y = tf.reshape(y, [-1, num_classes])      #  batch*frame_num x num_class

        W_multi_y = tf.tile( tf.reshape( tf.matmul(y, W_y), [FLAGS.batch_size, -1, 1]), [1, 1, frame_num] )
        V_multi_out = tf.transpose( tf.tile( tf.reshape( tf.matmul(outputs, V_y), [FLAGS.batch_size, -1, 1]), [1, 1, frame_num]  ), (0,2,1) )
        e_ = tf.tanh(W_multi_y + V_multi_out + b_y)  # batch x frame_num(t) x frame_num(j)

        alpha = tf.nn.softmax(e_, dim=2)
        outputs = tf.reshape(outputs, [FLAGS.batch_size, frame_num, -1])

        tf.summary.histogram('W_y', W_y)
        tf.summary.histogram('V_y', V_y)

        g_vector = tf.matmul(alpha, outputs)    # batch x frame_num x 2*num_hidden.  glimpse vector
        g_vector = tf.reshape(g_vector, [-1, 2 * FLAGS.num_hidden])

        dWy = tf.gradients(g_vector, W_y, name="W_y_gradients")
        tf.summary.histogram('W_y_gradients', dWy)

        W = tf.Variable(tf.truncated_normal ([2 * FLAGS.num_hidden, num_classes], stddev=0.1), name="w_dense") 
        b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b_dense")

        tf.summary.histogram('w_dense', W)

        self.logits = tf.matmul(g_vector, W) + b
            # Reshaping back to the original shape
        self.logits = tf.reshape(self.logits, [FLAGS.batch_size, -1, num_classes])
            # Time major
        self.logits = tf.transpose(self.logits, (1, 0, 2))




        
    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

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

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

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