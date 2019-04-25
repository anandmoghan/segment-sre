#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

name=swbd_sre_combined

num_gpu_jobs=20
max_gpu_jobs=6

window=100
hop=20

stage=0
nnet_dir=exp/xvector_nnet_1a

trials_segments=../save/trials/segments.lst

if [ ${stage} -le 0 ]; then
    python scripts/make_segments.py --window ${window} --hop ${hop} data/${name} data/${name}_segmented ${trials_segments} || exit
fi

if [ ${stage} -le 1 ]; then
    utils/fix_data_dir.sh data/${name}_segmented
    scripts/extract_segment_xvectors.sh --cmd "$train_cmd -q gpu.q -l hostname=compute-0-[0-4]" --nj ${num_gpu_jobs} --max-nj ${max_gpu_jobs} --use-gpu true \
        ${nnet_dir} data/${name}_segmented exp/xvectors_${name}_segmented
fi