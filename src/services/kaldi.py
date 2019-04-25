from subprocess import Popen, PIPE

import numpy as np
import re

from constants.kaldi import KALDI_QUEUE_FILE, KALDI_PATH_FILE


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd, decode=True, print_error=False):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, error = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        if error is not None:
            error = error.decode("utf-8")
            if print_error and not error == '':
                print(error)
        if decode:
            return output.decode("utf-8")
        return output

    def queue(self, cmd, queue_loc=KALDI_QUEUE_FILE, decode=True, print_error=True):
        cmd = '{} && ({} {})'.format(self.command, queue_loc, cmd)
        return self.run_command(cmd, decode, print_error)


def read_feats(scp_file, n_features, print_error=False):
    output = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file), print_error=print_error)
    features = re.split('\]', output)[:-1]
    utt_list = []
    feature_list = []
    for i, feature in enumerate(features):
        feature = re.split('\[', feature)
        utt_id = feature[0][:-1]
        feature = np.fromstring(feature[1][1:-1], dtype=float, sep=' \n').reshape([-1, n_features]).T
        utt_list.append(utt_id)
        feature_list.append(feature)
    return utt_list, feature_list


def read_vectors(scp_file, print_error=False):
    vectors = Kaldi().run_command('copy-vector scp:{} ark,t:'.format(scp_file), print_error=print_error)
    vectors = re.split('\]', vectors)[:-1]
    utt_list = []
    vector_list = []
    for vector in vectors:
        vector = re.split('\[', vector)
        utt_id = vector[0][:-1]
        vector = vector[1][1:-2]
        utt_list.append(utt_id)
        vector_list.append(np.fromstring(vector, np.float, sep=' '))
    return (utt_list, vector_list) if len(vector_list) > 1 else (utt_list[0], vector_list[0])

