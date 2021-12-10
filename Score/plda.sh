#!/bin/bash
# Yangwenhao 2019-12-16 20:27

#
# Plda score from Python training checkpoint
# 

if [ $# != 4 ]; then
  echo "Usage: plda.sh <data-path> <train-feat-dir> <test-feat-dir> <trials>"
  echo "e.g.: plda.sh data/train exp/train exp/test data/test/trials"
  echo "This script helps create srp to resample wav in wav.scp"
  exit 1
fi

data_dir=$1
train_feat_dir=$2
test_feat_dir=$3
trials=$4
# out_dir=$5
# model_path=SuResCNN10
logdir=$test_feat_dir/log

train_cmd="Vector_Score/run.pl --mem 8G"
lda_dim=200

# feat_dir=Data/checkpoint/${model}/soft/kaldi_feat
# data_dir=Data/dataset/voxceleb1/kaldi_feat/voxceleb1_test

#trials=$data_dir/trials
test_score=$test_feat_dir/scores_$(date "+%Y-%m-%d-%H:%M:%S")

$train_cmd $logdir/ivector-mean.log \
    ivector-mean scp:$train_feat_dir/xvectors.scp $test_feat_dir/mean.vec || exit 1;

$train_cmd $logdir/ivector-compute-lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim  "ark:ivector-subtract-global-mean scp:$train_feat_dir/xvectors.scp ark:- |" ark:$data_dir/utt2spk $train_feat_dir/transform.mat || exit 1;


# if ! [ -f $data_dir/spk2utt ];then
#     Score/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt
# fi

if ! [ -f $train_feat_dir/utt2spk ];then
    cat $train_feat_dir/xvectors.scp | awk '{print $1}' > $train_feat_dir/utt
    grep -f $train_feat_dir/utt $data_dir/utt2spk > $train_feat_dir/utt2spk

    Score/utt2spk_to_spk2utt.pl $train_feat_dir/utt2spk > $$train_feat_dir/spk2utt
fi


$train_cmd $logdir/ivector-compute-plda.log \
    ivector-compute-plda ark:$data_dir/spk2utt "ark:ivector-subtract-global-mean scp:$train_feat_dir/xvectors.scp ark:- | transform-vec $train_feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" $train_feat_dir/plda || exit 1;


$train_cmd $logdir/ivector-plda-scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $train_feat_dir/plda - |" \
    "ark:ivector-subtract-global-mean $train_feat_dir/mean.vec scp:$test_feat_dir/xvectors.scp ark:- | transform-vec $train_feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_feat_dir/mean.vec scp:$test_feat_dir/xvectors.scp ark:- | transform-vec $train_feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $test_score || exit 1;

eer=`compute-eer <(Score/prepare_for_eer.py $trials $test_score) 2> /dev/null`
mindcf1=`Score/compute_min_dcf.py --p-target 0.01 $test_score $trials 2> /dev/null`
mindcf2=`Score/compute_min_dcf.py --p-target 0.001 $test_score $trials 2> /dev/null`

test_result=$test_feat_dir/result_plda_$(date "+%Y-%m-%d-%H:%M:%S")

echo "EER: $eer%" >> $test_result
echo "minDCF(p-target=0.01): $mindcf1" >> $test_result
echo "minDCF(p-target=0.001): $mindcf2" >> $test_result

echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"



# ./Score/plda.sh /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/dev Data/xvector/ThinResNet34/vox2/klfb_egs_baseline/arcsoft_sgd_rop/chn32_Mean_basic_downNone_none1_SAP2_dp01_alpha0_em256_wde4_var/train/epoch_60 Data/xvector/ThinResNet34/vox2/klfb_egs_baseline/arcsoft_sgd_rop/chn32_Mean_basic_downNone_none1_SAP2_dp01_alpha0_em256_wde4_var/test_epoch_60_var
