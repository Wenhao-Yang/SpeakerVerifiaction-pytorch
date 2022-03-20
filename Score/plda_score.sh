#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plda_score.sh
# time: 2022/3/20 14:05
# Description: 

if [ $# != 5 ]; then
  echo "Usage: plda.sh <lda-dim> <data-path> <train-feat-dir> <test-feat-dir> <trials>"
  echo "e.g.: plda.sh data/train exp/train exp/test data/test/trials"
  echo "This script helps plda scoring for ivectors"
  exit 1
fi

lda_dim=$1
data_dir=$2
train_feat_dir=$3
test_feat_dir=$4
trials=$5
# out_dir=$5
# model_path=SuResCNN10
logdir=$test_feat_dir/log

transform_mat=$train_feat_dir/transform_dim${lda_dim}.mat
plda_model=$train_feat_dir/plda_${lda_dim}

#train_cmd="Score/run.pl --mem 8G"

test_score=$test_feat_dir/scores_${lda_dim}_$(date "+%Y-%m-%d-%H-%M-%S")

if ! [ -s $train_feat_dir/utt2spk ];then
    echo "Creating utt2spk!"
    cat $train_feat_dir/xvectors.scp | awk '{print $1}' > $train_feat_dir/utt
    grep -f $train_feat_dir/utt $data_dir/utt2spk > $train_feat_dir/utt2spk
fi

if ! [ -s $train_feat_dir/spk2utt ];then
    Score/utt2spk_to_spk2utt.pl ${train_feat_dir}/utt2spk > $train_feat_dir/spk2utt
    num_spks=`wc -l $train_feat_dir/spk2utt | awk '{print $1}'`
    echo "There are ${num_spks} speakers in train set!"
fi


if ! [ -s $transform_mat ];then
  python Score/Plda/compute_lda.py --total-covariance-factor=0.0 \
    --lda-dim $lda_dim  \
    --spk2utt $data_dir/spk2utt \
    --ivector-scp $train_feat_dir/xvectors.scp \
    --subtract-global-mean \
    --lda-mat $transform_mat
fi


if ! [ -s $plda_model ];then
    python Score/Plda/compute_plda.py --spk2utt $train_feat_dir/spk2utt \
      --ivector-scp $train_feat_dir/xvectors.scp \
      --mean-vec $train_feat_dir/mean.vec \
      --transform-vec $transform_mat \
      --plda-file $plda_model
fi

if ! [ -s $test_score ];then
  python Score/Plda/plda_scoring.py --normalize-length \
    --train-vec-scp $test_feat_dir/xvectors.scp \
    --test-vec-scp $test_feat_dir/xvectors.scp \
    --trials $trials \
    --plda-file $plda_model \
    --transform-vec $transform_mat \
    --score $test_score
fi

if [ -s $test_score ];then

  eer=`compute-eer <(Score/prepare_for_eer.py $trials $test_score) 2> /dev/null`
  mindcf1=`Score/compute_min_dcf.py --p-target 0.01 $test_score $trials 2> /dev/null`
  mindcf2=`Score/compute_min_dcf.py --p-target 0.001 $test_score $trials 2> /dev/null`

  test_result=$test_feat_dir/result_plda_${lda_dim}_$(date "+%Y.%m.%d.%H-%M-%S")

  echo "EER: $eer%" >> $test_result
  echo "minDCF(p-target=0.01) : $mindcf1" >> $test_result
  echo "minDCF(p-target=0.001): $mindcf2" >> $test_result

  echo "EER: $eer%"
  echo "minDCF(p-target=0.01) : $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi