import tensorflow as tf

from nn.layers.attention import context_attention


class SegmentAttention:
    def __init__(self, data_t, data_e, data_y, config):
        self.targets = data_y

        with tf.variable_scope('layer_1'):
            cell_fw = tf.contrib.rnn.GRUCell(int(config['layer_1_units']), activation=tf.tanh)
            cell_bw = tf.contrib.rnn.GRUCell(int(config['layer_1_units']), activation=tf.tanh)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, data_t, dtype=tf.float32)
            rnn_output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)

        with tf.variable_scope('layer_2'):
            cell_fw = tf.contrib.rnn.GRUCell(int(config['layer_2_units']), activation=tf.tanh)
            cell_bw = tf.contrib.rnn.GRUCell(int(config['layer_2_units']), activation=tf.tanh)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_output, dtype=tf.float32)
            rnn_output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
            pooled_output = context_attention(rnn_output, data_e, size=int(config['attention_size']))

        with tf.variable_scope('layer_3'):
            dense_output = tf.layers.dense(pooled_output, int(config['vector_dim']), activation=tf.tanh)
            dense_output = tf.concat([dense_output, data_e], axis=1)

        with tf.variable_scope('layer_4'):
            dense_output = tf.layers.dense(dense_output, int(config['layer_4_units']), activation=tf.tanh)

        with tf.variable_scope('output_layer'):
            self.scores = tf.layers.dense(dense_output, 1, activation=tf.nn.sigmoid)

        self.loss = tf.losses.mean_squared_error(data_y, self.scores)
        self.predicted = tf.greater(self.scores, 0.5)

        self.lr = tf.Variable(0.001, dtype=tf.float32, trainable=False, name='lr')
        self.global_step = tf.train.get_or_create_global_step()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def compute_loss(self, sess):
        return sess.run([self.loss, self.targets, self.predicted, self.global_step, self.scores])

    def get_global_step(self, sess):
        return sess.run(self.global_step)

    def train_step(self, lr, sess):
        loss, targets, predicted, global_step, _ = sess.run([self.loss, self.targets, self.predicted, self.global_step,
                                                             self.train_op], feed_dict={self.lr: lr})
        return loss, targets, predicted, global_step
