#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plt_det.sh
# time: 2021/9/1 15:46
# Description:

lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

stage=3
if [ $stage -le 0 ]; then
  python Eval/plt_det.py --score-name 英语,汉语 \
    --score-file Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var/score.2021.09.01.16\:24\:29,Data/xvector/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/aishell2_test_epoch_60_var/score.2021.09.01.16:35:39 \
    --save-path Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var \
    --pf-max 0.1
  exit
fi

if [ $stage -le 1 ]; then
  python Eval/plt_det.py --score-name cnceleb \
    --score-file Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/cnceleb_test_epoch_50_var/score.2021.09.01.15\:26\:52 \
    --save-path Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/cnceleb_test_epoch_50_var \
    --pf-max 0.1
fi

if [ $stage -le 2 ]; then
  num_spk=10
  distance=cos
  dataset=vox1
  subset=dev #test

  model_dir=Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist

#  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/baseline.pdf \
#    --hard-vector ${model_dir}_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors_2 \
#    --num-spk ${num_spk} --distance ${distance} \
#    --pca-dim 64

  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}_mani0_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/input.pdf \
    --num-spk ${num_spk} --distance ${distance} \
    --pca-dim 64 \
    --hard-vector ${model_dir}_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors_2
#
  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold.pdf \
    --num-spk ${num_spk} --distance ${distance} \
    --pca-dim 64 \
    --hard-vector ${model_dir}_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors_2
  exit
fi

if [ $stage -le 3 ]; then
  num_spk=20
  distance=cos
  dataset=vox1
  subset=dev #test

  model_dir=Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist_mani023_lamda2.0

  python -W ignore Eval/filter_vectors.py --score-file ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/score.2022.10.24.21:49:19 \
    --trials ${lstm_dir}/data/vox1/dev/trials_hard \
    --threshold 0.2984 \
    --output-file ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors_3 \
    --confidence-interval 0.1

#  python -W ignore Eval/plt_tsne.py --scp-file Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist_mani0_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold0.v1.pdf \
#    --num-spk ${num_spk} --distance ${distance}
#
#  python -W ignore Eval/plt_tsne.py --scp-file Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold023.v1.pdf \
#    --num-spk ${num_spk} --distance ${distance}
  exit
fi

if [ $stage -le 4 ]; then
  num_spk=10
  distance=cos
  dataset=vox1
  subset=test #test

  model_dir=Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist

  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/baseline.pdf \
    --hard-vector ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors \
    --num-spk ${num_spk} --distance ${distance} \
    --pca-dim 64

  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}_mani0_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/input.pdf \
    --num-spk ${num_spk} --distance ${distance} \
    --pca-dim 64 \
    --hard-vector ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors
#
  python -W ignore Eval/plt_tsne.py --scp-file ${model_dir}_mani023_lamda2.0/123456/${dataset}_${subset}_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold.pdf \
    --num-spk ${num_spk} --distance ${distance} \
    --pca-dim 64 \
    --hard-vector ${model_dir}/123456/${dataset}_${subset}_var/testwidth1.000000/hard_vectors
  exit
fi
