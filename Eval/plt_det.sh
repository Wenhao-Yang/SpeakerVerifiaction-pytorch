#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plt_det.sh
# time: 2021/9/1 15:46
# Description:

stage=1
if [ $stage -le 0 ]; then
  python Eval/plt_det.py --score-name vox1,aishell \
    --score-file Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var/score.2021.09.01.15\:25\:51,Data/xvector/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/aishell2_test_epoch_60_var/score.2021.09.01.16:35:39 \
    --save-path Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var \
    --pf-max 0.1

fi

if [ $stage -le 1 ]; then
  python Eval/plt_det.py --score-name cnceleb \
    --score-file Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/cnceleb_test_epoch_50_var/score.2021.09.01.15\:26\:52 \
    --save-path Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/cnceleb_test_epoch_50_var \
    --pf-max 0.1

fi
