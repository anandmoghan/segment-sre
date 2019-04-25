#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.

# Begin configuration section.
nj=30
max_nj=10
cmd="run.pl"
chunk_size=100 # The chunk size over which the embedding is extracted.
use_gpu=false
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --max-nj <n|10>                                  # Max number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

src_dir=$1
data=$2
dir=$3

for f in ${src_dir}/final.raw ${data}/feats.scp; do
  [ ! -f ${f} ] && echo "No such file $f" && exit 1;
done

nnet=${src_dir}/final.raw
if [ -f ${src_dir}/extract.config ] ; then
  echo "$0: using $src_dir/extract.config to extract segment xvectors"
  nnet="nnet3-copy --nnet-config=${src_dir}/extract.config ${src_dir}/final.raw - |"
fi

mkdir -p ${dir}/log

utils/split_data.sh ${data} ${nj}
echo "$0: extracting segment xvectors for $data"
sdata=${data}/split${nj}/JOB

# Set up the features
feat="ark:copy-feats scp:${sdata}/feats.scp ark:- |"

if [ ${stage} -le 0 ]; then
  echo "$0: extracting segment xvectors from nnet"
  if ${use_gpu}; then
    for g in $(seq ${nj}); do
      ${cmd} --gpu 1 ${dir}/log/extract.${g}.log \
        nnet3-xvector-compute --use-gpu=yes --min-chunk-size=${chunk_size} --chunk-size=${chunk_size} \
        "$nnet" "`echo ${feat} | sed s/JOB/${g}/g`" ark,scp:${dir}/xvector.${g}.ark,${dir}/xvector.${g}.scp || exit 1 &
    done
    wait
  else
    ${cmd} JOB=1:${nj} ${dir}/log/extract.JOB.log \
      nnet3-xvector-compute --use-gpu=no --min-chunk-size=${chunk_size} --chunk-size=${chunk_size} \
      "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
  fi
fi

if [ ${stage} -le 1 ]; then
  echo "$0: combining segment xvectors across jobs"
  for j in $(seq ${nj}); do cat ${dir}/xvector.${j}.scp; done > ${dir}/xvector.scp || exit 1;
fi
