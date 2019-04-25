import numpy as np
import tensorflow as tf

import constants.common as const


class EarlyStopping:
    def __init__(self, max_to_keep=5):
        self.num_models = max_to_keep
        self.val_losses = np.ones([max_to_keep, 1]) * np.iinfo(np.int).max
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.should_stop = False

    def update(self, val_loss, model_path, sess):
        idx = np.argmax(self.val_losses)
        if self.val_losses[idx] > val_loss:
            self.val_losses[idx] = val_loss
            self.saver.save(sess, model_path, latest_filename=const.LATEST_MODEL)
        else:
            self.should_stop = True
