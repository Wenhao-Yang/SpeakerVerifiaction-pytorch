#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plt_det.sh
# time: 2021/9/1 15:46
# Description:

stage=2
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
  num_spk=20
  distance=cos
  python Eval/plt_tsne.py --scp-file Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist/123456/vox1_test_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/baseline.v1.pdf \
    --num-spk ${num_spk} --distance ${distance}

  python Eval/plt_tsne.py --scp-file Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist_mani0_lamda2.0/123456/vox1_test_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold0.v1.pdf \
    --num-spk ${num_spk} --distance ${distance}

  python Eval/plt_tsne.py --scp-file Data/xvector/ThinResNet34/vox1/wave_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_var2ses_bashuf2_dist_mani023_lamda2.0/123456/vox1_test_var/testwidth1.000000/xvectors.scp --out-pdf Misc/data/manifold023.v1.pdf \
    --num-spk ${num_spk} --distance ${distance}

fi

