import configparser as cp
from os.path import join

import numpy as np
import tensorflow as tf

from models.get_model import all_models
from services.common import print_log
from services.data import segment_batch_generator
from services.gpu import set_gpu

import constants.common as const
from services.score import ScoreHolder
from services.train import EarlyStopping

config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('./train.cfg')
except IOError:
    raise IOError('Oh dear! Something is wrong with the config file.')

print_log('Fetching trials')
train_trials = np.genfromtxt(config[const.FILES][const.TRAIN_TRIALS], dtype=str)
val_trials = np.genfromtxt(config[const.FILES][const.VAL_TRIALS], dtype=str)

print_log('Fetching segment xvectors')
segment_xvectors = np.genfromtxt(config[const.FILES][const.SEGMENT_XVECTORS], dtype=str)
segment_dict = dict(zip(segment_xvectors[:, 0], segment_xvectors[:, 1]))

print_log('Fetching xvectors')
xvectors = np.genfromtxt(config[const.FILES][const.XVECTORS], dtype=str)
enroll_dict = dict(zip(xvectors[:, 0], xvectors[:, 1]))


def train_generator():
    return segment_batch_generator(train_trials, segment_dict, enroll_dict, config)


def val_generator():
    return segment_batch_generator(val_trials, segment_dict, enroll_dict, config)


print_log('Initialising model')
set_gpu(-1, min_free_fraction=0.67)

output_types = (tf.float32, tf.float32, tf.int32)
output_shapes = (tf.TensorShape([None, None, int(config[const.MODEL][const.VECTOR_DIM])]),
                 tf.TensorShape([None, int(config[const.MODEL][const.VECTOR_DIM])]),
                 tf.TensorShape([None, 1]))
train_data_set = tf.data.Dataset.from_generator(train_generator, output_types, output_shapes) \
    .prefetch(int(config[const.TRAIN][const.PREFETCH]))
val_data_set = tf.data.Dataset.from_generator(val_generator, output_types, output_shapes) \
    .prefetch(int(config[const.TRAIN][const.PREFETCH]))

data_iterator = tf.data.Iterator.from_structure(train_data_set.output_types, train_data_set.output_shapes)
train_iterator = data_iterator.make_initializer(train_data_set)
val_iterator = data_iterator.make_initializer(val_data_set)
data_t, data_e, data_y = data_iterator.get_next()

model = all_models[config[const.MODEL][const.MODEL_ID]](data_t, data_e, data_y, config[const.MODEL])

print_log('Starting training')
train_score = ScoreHolder()
val_score = ScoreHolder()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
model_path = join(config[const.PATHS][const.MODEL_LOC], 'model')
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    early_stop = EarlyStopping()
    lr_decay = (float(config[const.TRAIN][const.FINAL_LR]) / float(config[const.TRAIN][const.INITIAL_LR])) ** (1 / int(config[const.TRAIN][const.EPOCHS]))
    for e in range(int(config[const.TRAIN][const.EPOCHS])):
        print_log('Epoch: {}'.format(e + 1))
        current_lr = float(config[const.TRAIN][const.INITIAL_LR]) * (lr_decay ** e)
        sess.run(train_iterator)
        train_score.reset()
        try:
            while True:
                loss, targets, predicted, global_step = model.train_step(current_lr, sess)
                train_score.add(targets, predicted, loss)
                if global_step % int(config[const.TRAIN][const.PRINT_STEP]) == 0:
                    print_log('Epoch: {} | Global Step: {} | Loss: {:.3f} | Accuracy: {:.2f} | Miss: {:.3f} | False Alarm: {:.3f}'
                              .format(e + 1, global_step, train_score.batch_loss(), train_score.batch_accuracy(),
                                      train_score.batch_miss(), train_score.batch_false_alarm()))
                    train_score.batch_start()
        except tf.errors.OutOfRangeError:
            pass

        print_log('Epoch: {} | Train Summary\n\t\t\t\tLoss: {:.3f} | Accuracy: {:.2f} | Miss: {:.3f} | False Alarm: {:.3f}'
                  .format(e + 1, train_score.loss(), train_score.accuracy(), train_score.miss(), train_score.false_alarm()))

        sess.run(val_iterator)
        val_score.reset()
        try:
            while True:
                loss, targets, predicted, global_step, _ = model.compute_loss(sess)
                val_score.add(targets, predicted, loss)
        except tf.errors.OutOfRangeError:
            pass

        print_log('Epoch: {} | Validation Summary:\n\t\t\t\tLoss: {:.3f} | Accuracy: {:.2f} | Miss: {:.3f} | False Alarm: {:.3f}'
                  .format(e + 1, val_score.loss(), val_score.accuracy(), val_score.miss(), val_score.false_alarm()))

        early_stop.update(val_score.loss(), model_path, sess)
        if early_stop.should_stop:
            break
