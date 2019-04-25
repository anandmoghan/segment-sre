from os import remove
from os.path import join

import numpy as np

import constants.common as const
from services.kaldi import read_vectors


def segment_unit_generator(trials, segments_dict, enroll_dict, config):
    np.random.shuffle(trials)
    hop = int(config[const.MODEL][const.HOP])
    window = int(config[const.MODEL][const.WINDOW])
    for idx, (xid, start, end, spk, yid, other_spk) in enumerate(trials):
        tmp_scp = join(config[const.PATHS][const.TMP_LOC], 'read_{}_{}.scp'.format(config[const.TRAIN][const.MODEL_TAG], idx))
        try:
            with open(tmp_scp, 'w') as f:
                start_frame = int(start)
                while start_frame <= int(end) - window:
                    end_frame = start_frame + window
                    uid = '{}_{:06d}-{:06d}'.format(xid, start_frame, end_frame)
                    f.write('{} {}\n'.format(uid, join(config[const.PATHS][const.SAVE_LOC], segments_dict[uid])))
                    start_frame = end_frame - hop
                f.write('{} {}\n'.format(xid, join(config[const.PATHS][const.SAVE_LOC], enroll_dict[xid])))

            _, vectors = read_vectors(tmp_scp)
            remove(tmp_scp)
            yield np.array(vectors[:-1]), np.array(vectors[-1]), np.array([int(spk == other_spk)])
        except KeyError:
            pass


def segment_batch_generator(trials, segments_dict, enroll_dict, config):
    np.random.shuffle(trials)
    hop = int(config[const.MODEL][const.HOP])
    window = int(config[const.MODEL][const.WINDOW])
    batch_size = int(config[const.TRAIN][const.BATCH_SIZE])
    num_batches = int(len(trials) / batch_size)
    for idx, batch_trials in enumerate(np.array_split(trials, num_batches)):
        tmp_scp = join(config[const.PATHS][const.TMP_LOC], 'read_{}_{}.scp'.format(config[const.TRAIN][const.MODEL_TAG], idx))
        segments = []
        xvectors = []
        labels = []
        batch_size = len(batch_trials)
        for xid, start, end, spk, yid, other_spk in batch_trials:
            tmp_segments = []
            try:
                start_frame = int(start)
                while start_frame <= int(end) - window:
                    end_frame = start_frame + window
                    uid = '{}_{:06d}-{:06d}'.format(xid, start_frame, end_frame)
                    tmp_segments.append('{} {}\n'.format(uid, join(config[const.PATHS][const.SAVE_LOC], segments_dict[uid])))
                    start_frame = end_frame - hop
                xvectors.append('{} {}\n'.format(xid, join(config[const.PATHS][const.SAVE_LOC], enroll_dict[xid])))
                segments += tmp_segments
                labels.append(int(spk == other_spk))
            except KeyError:
                batch_size -= 1

        if batch_size > 0:
            with open(tmp_scp, 'w') as f:
                f.writelines(segments)
                f.writelines(xvectors)

            _, vectors = read_vectors(tmp_scp)
            yield np.array(np.array_split(vectors[:-batch_size], batch_size)), np.array(vectors[-batch_size:]), np.array(labels).reshape([-1, 1])
