from os.path import join
from sklearn.model_selection import train_test_split

import numpy as np

from services.common import print_log

trials_loc = '../save/trials'
segments_list = join(trials_loc, 'segments.lst')
train_segments_list = join(trials_loc, 'train_segments.lst')
val_segments_list = join(trials_loc, 'val_segments.lst')

print_log('Parsing {}'.format(segments_list))
spk2utt = dict()
with open(segments_list) as f:
    for line in f.readlines():
        tokens = line.strip().split()
        try:
            utt = spk2utt[tokens[3]]
        except KeyError:
            utt = []
        utt.append(line)
        spk2utt[tokens[3]] = utt

print_log('Splitting speakers')
train_speakers, val_speakers = train_test_split(list(spk2utt.keys()), test_size=0.05)

print_log('Writing to files')
for speakers, segments_file in zip([train_speakers, val_speakers], [train_segments_list, val_segments_list]):
    with open(segments_file, 'w') as f:
        for s in speakers:
            f.write('{}'.format(''.join(spk2utt[s])))
