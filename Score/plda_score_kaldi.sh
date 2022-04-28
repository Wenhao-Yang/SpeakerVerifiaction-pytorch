#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plda_score_python.sh
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
adaptation=false
# out_dir=$5
# model_path=SuResCNN10
logdir=$test_feat_dir/log

transform_mat=$train_feat_dir/transform_dim${lda_dim}.mat
plda_model=$train_feat_dir/plda_${lda_dim}

train_cmd="Score/run.pl --mem 8G"

test_score=$test_feat_dir/scores_${lda_dim}_$(date "+%Y-%m-%d-%H-%M-%S")

stage=0

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

if [ $stage -le 0 ]; then
#if ! [ -s $train_feat_dir/mean.vec ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $logdir/compute_mean.log \
    ivector-mean scp:$train_feat_dir/xvectors.scp \
    $train_feat_dir/mean.vec || exit 1;
fi

  # This script uses LDA to decrease the dimensionality prior to PLDA.
if ! [ -s $transform_mat ];then
  echo "Computing LDA transform matrix ..."

  $train_cmd $logdir/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_feat_dir/xvectors.scp ark:- |" \
    ark:$train_feat_dir/utt2spk $transform_mat || exit 1;
fi

if ! [ -s $plda_model ];then
  echo "Computing PLDA stats ..."
  # Train the PLDA model.
  # subtract global mean and do lda transform before PLDA classification
  $train_cmd $logdir/plda.log \
    ivector-compute-plda ark:$train_feat_dir/spk2utt \
    "ark:ivector-subtract-global-mean scp:$train_feat_dir/xvectors.scp ark:- | transform-vec $transform_mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $plda_model || exit 1;
fi

# Adaptation plda using out-of-domain dataset
if $adaptation; then
  echo "Adapting PLDA ..."
  ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $plda_model \
    "ark:ivector-subtract-global-mean scp:$test_feat_dir/xvectors.scp ark:- | transform-vec $transform_mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${plda_model}_adapt || exit 1

  mv ${plda_model} ${plda_model}_noadapt
  mv ${plda_model}_adapt ${plda_model}
fi

if ! [ -s $test_score ];then
  echo "Scoring with PLDA ..."
  $train_cmd $logdir/test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $plda_model - |" \
    "ark:ivector-subtract-global-mean $train_feat_dir/mean.vec scp:$test_feat_dir/xvectors.scp ark:- | transform-vec $transform_mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_feat_dir/mean.vec scp:$test_feat_dir/xvectors.scp ark:- | transform-vec $transform_mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $test_score || exit 1;
fi

if [ -s $test_score ];then
  echo "Computing EER minDCF ..."
  eer=`compute-eer <(Score/prepare_for_eer.py $trials $test_score) 2> /dev/null`
  mindcf1=`Score/compute_min_dcf.py --p-target 0.01 $test_score $trials 2> /dev/null`
  mindcf2=`Score/compute_min_dcf.py --p-target 0.001 $test_score $trials 2> /dev/null`

  test_result=$test_feat_dir/result_plda_${lda_dim} #_$(date "+%Y.%m.%d.%H-%M-%S")

  echo -e "\n$(date "+%Y.%m.%d.%H-%M-%S"):\nEER: $eer%" >> $test_result
  echo "minDCF(p-target=0.01) : $mindcf1" >> $test_result
  echo "minDCF(p-target=0.001): $mindcf2" >> $test_result

  echo -e "\nEER: $eer%"
  echo "minDCF(p-target=0.01) : $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi