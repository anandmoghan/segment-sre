from os.path import join

import numpy as np
import sys

from services.common import print_log, run_parallel

# TODO: Edit this file to make trials before extracting x vectors

try:
    selection = sys.argv[1]
except IndexError:
    selection = 'train'

print_log('Making trials for: {}'.format(selection))
save_loc = '../save'
segment_file = join(save_loc, 'trials/{}_segments.lst'.format(selection))
trials_file = join(save_loc, 'trials/{}_trials.lst'.format(selection))

sample_len = 1000
max_trials = 2000000 if selection == 'train' else 20000
num_workers = 20

print_log('Reading segments list')
segments = np.genfromtxt(segment_file, dtype=str)
utt = list(zip(segments[:, 0], segments[:, 1].astype(int), segments[:, 1].astype(int) + segments[:, 2].astype(int), segments[:, 3]))
utt2spk = list(zip(segments[:, 0], segments[:, 3]))


def get_trials(items):
    utt_list, count, wid = items
    half_count = int(count / 2)

    samples_idx = np.random.choice(len(utt_list), half_count, replace=True)
    utt_list = utt_list[samples_idx]
    worker_trials = []
    for i, (uid, s, e, sp) in enumerate(utt_list):
        same_spk = False
        different_spk = False
        while not (same_spk and different_spk):
            sampled_idx = np.random.choice(len(utt2spk), 1)
            sampled_utt, sampled_spk = utt2spk[sampled_idx[0]]
            if not same_spk and sp == sampled_spk:
                worker_trials.append([uid, s, e, sp, sampled_utt, sampled_spk])
                same_spk = True
            elif not different_spk and not sp == sampled_spk:
                worker_trials.append([uid, s, e, sp, sampled_utt, sampled_spk])
                different_spk = True
        if (i + 1) % half_count == 0:
            print_log('Worker {}: Made 50% trials'.format(wid))

    # print_log('Worker {}: Made {} trials'.format(wid, len(worker_trials)))
    return worker_trials


print_log('Making trials')
trials_per_worker = [int(max_trials / num_workers)] * num_workers
worker_ids = range(0, num_workers)
utt_split = np.array_split(utt, num_workers)
trials_split = run_parallel(get_trials, list(zip(utt_split, trials_per_worker, worker_ids)))

print_log('Writing to {}'.format(trials_file))
with open(trials_file, 'w') as f:
    for trials in trials_split:
        for t in trials:
            f.write('{}\n'.format(' '.join(t)))
