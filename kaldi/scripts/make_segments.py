from os.path import exists, join as join_path
from os import makedirs

import argparse as ap
import numpy as np
import re
import sys

from tqdm import tqdm

script_file = sys.argv[0]

parser = ap.ArgumentParser()
parser.add_argument('--hop', type=int, default=20, help='Number of hop frames')
parser.add_argument('--max-samples', type=int, default=3, help='Max samples from a file')
parser.add_argument('--segment-length', type=int, default=1000, help='Segment length from a file')
parser.add_argument('--step', type=int, default=500, help='Step size')
parser.add_argument('--window', type=int, default=100, help='Number of window frames')
parser.add_argument('data_dir', action="store")
parser.add_argument('new_data_dir', action="store")
parser.add_argument('trial_segments_file', action="store")
args = parser.parse_args()


def make_directory(path):
    if not exists(path):
        makedirs(path)


def make_file_dict(file, pattern='[\s]+'):
    file_dict = dict()
    with open(file) as f:
        for line in f.readlines():
            tokens = re.split(pattern, line.strip())
            file_dict[tokens[0]] = tokens[1]
    return file_dict


utt2num_frames = join_path(args.data_dir, 'utt2num_frames')
utt2spk = join_path(args.data_dir, 'utt2spk')
feats_scp = join_path(args.data_dir, 'feats.scp')

if not exists(utt2num_frames):
    print('{}: utt2num_frames file missing'.format(script_file))
    exit(1)

if not exists(utt2spk):
    print('{}: utt2spk file missing'.format(script_file))
    exit(1)

if not exists(feats_scp):
    print('{}: feats.scp file missing'.format(script_file))
    exit(1)

make_directory(args.new_data_dir)

utt2num_frames_dict = make_file_dict(utt2num_frames)
utt2spk_dict = make_file_dict(utt2spk)
feats_scp_dict = make_file_dict(feats_scp)

new_feats_scp = open(join_path(args.new_data_dir, 'feats.scp'), 'w')
new_utt2spk = open(join_path(args.new_data_dir, 'utt2spk'), 'w')
trial_segments = open(args.trial_segments_file, 'w')

print('{}: Making segments'.format(script_file))
p_bar = tqdm(total=len(utt2num_frames_dict.keys()))
for utt, frames in list(utt2num_frames_dict.items()):
    try:
        frames = int(frames)
        if frames >= args.segment_length:
            sample_heads = np.arange(0, frames - args.segment_length, args.step)
            num_samples = args.max_samples if args.max_samples <= len(sample_heads) else len(sample_heads)
            starts = np.random.choice(sample_heads, num_samples, replace=False)
            scp = feats_scp_dict[utt]
            spk = utt2spk_dict[utt]
            for start_frame in starts:
                trial_segments.write('{} {} {} {}\n'.format(utt, start_frame, args.segment_length, spk))
                final_frame = start_frame + args.segment_length - args.window
                while start_frame <= final_frame:
                    end_frame = start_frame + args.window
                    new_utt2spk.write('{}_{:06d}-{:06d} {}\n'.format(utt, start_frame, end_frame, spk))
                    new_feats_scp.write('{}_{:06d}-{:06d} {}[{}:{}]\n'.format(utt, start_frame, end_frame, scp,
                                                                              start_frame, end_frame - 1))
                    start_frame = end_frame - args.hop
    except KeyError:
        pass
    p_bar.update(1)

p_bar.close()

print('{}: Writing to {}'.format(script_file, args.new_data_dir))
new_feats_scp.close()
new_utt2spk.close()
trial_segments.close()
