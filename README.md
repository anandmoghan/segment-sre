# Segment Level Speaker Recognition
Uses 1 sec. xvector segments as input features to a BLSTM network with uses enrollment xvectors to focus attention on the test and do a binary classification.

## Instructions
  1. Create segments with `kaldi/extract_segments.sh`
  2. Sample train and validation trials with `src/make_trials.py' from the above created segments
  3. Make necessary modifications to `src/train.cfg` and run `src/train.py`
