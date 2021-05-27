#!/bin/bash

stage=10
# global variale
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

nnet_dir=data
# reusme options
feat_type=spect
feat=log
loss=arcsoft
model=TDNN_v5
encod=None
#dataset=aishell2
dataset=vox2
test_set=cnceleb

# extract options
#xvector_dir=Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4/vox1_test_var/xvectors/epoch_40
xvector_dir=Data/xvector/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/vox1_test_var/xvectors/epoch_60
train_xvector_dir=${xvector_dir}/train

xvector_dir=Data/xvector/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/cnceleb_test_var/xvectors/epoch_60
test_xvector_dir=${xvector_dir}/test

# test options
adaptation=false

#test_trials=${lstm_dir}/data/vox1/pyfb/test_fb40/trials
#train_dir=${lstm_dir}/data/vox1/egs/pyfb/dev_fb40

test_trials=${lstm_dir}/data/${test_set}/${feat_type}/test_${feat}/trials
train_dir=${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}
#data_dir=${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}

if [ $stage -le 9 ]; then

  for subset in dev eval; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --input-length var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/${test_set}_${subset}_epoch_60_var \
      --resume Data/checkpoint/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  ivector-mean scp:$train_xvector_dir/xvectors.scp $train_xvector_dir/mean.vec || exit 1
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=500
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_xvector_dir/xvectors.scp ark:- |" \
    ark:$train_dir/utt2spk $train_xvector_dir/transform.mat || exit 1

  # Train the PLDA model.
  # subtract global mean and do lda transform before PLDA classification
  ivector-compute-plda ark:$train_dir/spk2utt \
    "ark:ivector-subtract-global-mean scp:$train_xvector_dir/xvectors.scp ark:- | transform-vec $train_xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $train_xvector_dir/plda || exit 1

  # Adaptation plda using out-of-domain dataset
  if $adaptation; then
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $train_xvector_dir/plda \
      "ark:ivector-subtract-global-mean scp:$test_xvector_dir/xvectors.scp ark:- | transform-vec $train_xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      $train_xvector_dir/plda_adapt || exit 1

    mv $train_xvector_dir/plda $train_xvector_dir/plda_nada
    mv $train_xvector_dir/plda_adapt $train_xvector_dir/plda
  fi
fi

if [ $stage -le 11 ]; then

  if $adaptation; then
    scores_file=adapt_scores_test
  else
    scores_file=scores_test
  fi
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $train_xvector_dir/plda - |" \
    "ark:ivector-subtract-global-mean $train_xvector_dir/mean.vec scp:$test_xvector_dir/xvectors.scp ark:- | transform-vec $train_xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $train_xvector_dir/mean.vec scp:$test_xvector_dir/xvectors.scp ark:- | transform-vec $train_xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$test_trials' | cut -d\  --fields=1,2 |" $test_xvector_dir/$scores_file || exit 1
fi

if [ $stage -le 12 ]; then
  eer=$(compute-eer <(Score/prepare_for_eer.py $test_trials $test_xvector_dir/scores_test) 2>/dev/null)
  mindcf1=$(Score/compute_min_dcf.py --p-target 0.01 $test_xvector_dir/scores_test $test_trials 2>/dev/null)
  mindcf2=$(Score/compute_min_dcf.py --p-target 0.001 $test_xvector_dir/scores_test $test_trials 2>/dev/null)
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

# 20210527
# kaldi plda
# Data/xvector/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/vox1_test_var/xvectors/epoch_60

# vox1 test
# dim=200
# EER: 4.38%
#minDCF(p-target=0.01): 0.4245
#minDCF(p-target=0.001): 0.5548

# adaptation dim=400
# EER: 4.38%
# minDCF(p-target=0.01): 0.4245
# minDCF(p-target=0.001): 0.5548

# no adaptation dim=500
# EER: 3.876%
# minDCF(p-target=0.01): 0.3538
# minDCF(p-target=0.001): 0.4406
