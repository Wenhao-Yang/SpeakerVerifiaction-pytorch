stage=9

nnet_dir=data

feat_type=spect
feat=log
loss=arcsoft
model=TDNN_v5
encod=None
dataset=aishell2
test_set=sitw
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 79 ]; then

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
  xvector_dir=Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4/vox1_test_var/xvectors/epoch_40/train
  train_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/egs/pyfb/dev_fb40
  data_dir=${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}

  ivector-mean scp:$xvector_dir/xvectors.scp $xvector_dir/mean.vec || exit 1

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$xvector_dir/xvectors.scp ark:- |" \
    ark:$train_dir/utt2spk $xvector_dir/transform.mat || exit 1

  # Train the PLDA model.
  # subtract global mean and do lda transform before PLDA classification
  ivector-compute-plda ark:$train_dir/spk2utt \
    "ark:ivector-subtract-global-mean scp:$xvector_dir/xvectors.scp ark:- | transform-vec $xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $xvector_dir/plda || exit 1
fi

if [ $stage -le 11 ]; then
  test_xvector_dir=Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4/vox1_test_var/xvectors/epoch_40/test
  test_trials=/home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pyfb/test_fb40/trials
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $xvector_dir/plda - |" \
    "ark:ivector-subtract-global-mean $xvector_dir/mean.vec scp:$test_xvector_dir/xvectors.scp ark:- | transform-vec $xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $xvector_dir/mean.vec scp:$test_xvector_dir/xvectors.scp ark:- | transform-vec $xvector_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$test_trials' | cut -d\  --fields=1,2 |" $test_xvector_dir/scores_voxceleb1_test || exit 1
fi

if [ $stage -le 12 ]; then
  eer=$(compute-eer <(Score/prepare_for_eer.py $test_trials $test_xvector_dir/scores_voxceleb1_test) 2>/dev/null)
  mindcf1=$(Score/compute_min_dcf.py --p-target 0.01 $test_xvector_dir/scores_voxceleb1_test $test_trials 2>/dev/null)
  mindcf2=$(Score/compute_min_dcf.py --p-target 0.001 $test_xvector_dir/scores_voxceleb1_test $test_trials 2>/dev/null)
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
