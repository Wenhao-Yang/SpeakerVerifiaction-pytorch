#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: plt_det.sh
# time: 2021/9/1 15:46
# Description:

stage=0
if [ $stage -le 0 ]; then
  python Eval/plt_det.py --score-name vox1,cnceleb \
    --score-file Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var/score.2021.09.01.15\:25\:51,Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/cnceleb_test_epoch_50_var/score.2021.09.01.15\:26\:52 \
    --save-path Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/vox1_test_epoch_50_var \
    --pf-max 0.1


fi