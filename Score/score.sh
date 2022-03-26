#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plda_score_python.sh
# time: 2022/3/20 14:05
# Description:

stage=10

lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification/data

if [ $stage -le 0 ]; then
  trials=$voxceleb1_trials
  scores=exp/scores_voxceleb1_test
  eer=$(compute-eer <(Vector_Score/prepare_for_eer.py $trials $scores) 2>/dev/null)
  mindcf1=$(Vector_Score/compute_min_dcf.py --p-target 0.01 $scores $trials 2>/dev/null)
  mindcf2=$(Vector_Score/compute_min_dcf.py --p-target 0.001 $scores $trials 2>/dev/null)
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

if [ $stage -le 5 ]; then
  python compute_mean.py
  python compute_lda.py
  python compute_plda.py
  python plda_scoring.py
fi


if [ $stage -le 10 ]; then
  xvector_dir=Data/xvector/ThinResNet18/cnceleb/klfb_egs_baseline/arcdist_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_lrmaxmargin1_wd5e4_var/cnceleb_test_fix/xvectors_a/epoch_60
  train_vec_dir=${xvector_dir}/train
  test_vec_dir=${xvector_dir}/test
  data_dir=${lstm_dir}/cnceleb/klfb/dev_fb40
  trials=${lstm_dir}/cnceleb/klfb/test_fb40/trials

  lda_dim=512

  ./Score/plda_score_kaldi.sh ${lda_dim} ${data_dir} ${train_vec_dir} ${test_vec_dir} ${trials}
  exit
fi

if [ $stage -le 20 ]; then
  xvector_dir=Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_basic_STAP_em512_wd5e4_var/cnceleb_test_var/xvectors_a/epoch_50
  train_vec_dir=${xvector_dir}/train
  test_vec_dir=${xvector_dir}/test
  data_dir=${lstm_dir}/cnceleb/klfb/dev_fb40
  trials=${lstm_dir}/cnceleb/klfb/test_fb40/trials

  lda_dim=200

  ./Score/plda_score_kaldi.sh ${lda_dim} ${data_dir} ${train_vec_dir} ${test_vec_dir} ${trials}

fi