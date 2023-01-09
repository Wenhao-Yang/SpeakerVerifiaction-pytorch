#!/usr/bin/env bash

stage=200
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

# ===============================    LoResNet10    ===============================
if [ $stage -le 0 ]; then
  for loss in asoft soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"

    python TrainAndTest/test_sitw.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --veri-pairs 12800 \
      --loss-type ${loss} \
      --gpu-id 0 \
      --epochs 20
  done
fi

if [ $stage -le 5 ]; then
  model=LoResNet10
  #  --resume Data/checkpoint/LoResNet10/spect/${loss}_dp25_128/checkpoint_24.pth \
  #  for loss in soft ; do
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_spect/dev_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_spect/all_wcmvn \
      --nj 12 \
      --model ${model} \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_wcmvn/checkpoint_24.pth \
      --xvector-dir Data/xvector/LoResNet10/spect/${loss}_wcmvn \
      --loss-type ${loss} \
      --trials trials \
      --num-valid 0 \
      --gpu-id 0
  done

#  for loss in center ; do
fi

if [ $stage -le 6 ]; then
  model=LoResNet10
  #  --resume Data/checkpoint/LoResNet10/spect/${loss}_dp25_128/checkpoint_24.pth \
  #  for loss in soft ; do
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"

    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_spect/dev_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --model ${model} \
      --channels 64,128,256,256 \
      --resnet-size 18 \
      --extract \
      --kernel-size 3,3 \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet18/spect/soft_dp25/checkpoint_24.pth \
      --xvector-dir Data/xvector/LoResNet18/spect/soft_dp05 \
      --loss-type ${loss} \
      --trials trials.backup \
      --num-valid 0 \
      --gpu-id 0
  done

fi

# ===============================    ExResNet    ===============================

if [ $stage -le 7 ]; then
  model=ExResNet
  datasets=vox1
  feat=fb64_3w
  loss=soft
  for encod in SAP SASP STAP None; do
    echo -e "\n\033[1;4;31m Test ${model}_${encod} with ${loss}\033[0m\n"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/dev_${feat} \
      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/test_${feat} \
      --nj 12 \
      --model ${model} \
      --resnet-size 10 \
      --remove-vad \
      --kernel-size 5,5 \
      --embedding-size 128 \
      --resume Data/checkpoint/${model}10/${datasets}_${encod}/${feat}/${loss}/checkpoint_24.pth \
      --xvector-dir Data/xvector/${model}10/${datasets}_${encod}/${feat}/${loss} \
      --loss-type ${loss} \
      --trials trials \
      --num-valid 2 \
      --gpu-id 0
  done
fi

# ===============================    TDNN    ===============================

#stage=200
if [ $stage -le 15 ]; then
  model=TDNN
  #  feat=fb40
  #  for loss in soft ; do
  #    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
  #    python TrainAndTest/test_egs.py \
  #      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_no_sil \
  #      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_no_sil \
  #      --nj 12 \
  #      --model ${model} \
  #      --embedding-size 128 \
  #      --feat-dim 40 \
  #      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_18.pth
  #      --loss-type soft \
  #      --num-valid 2 \
  #      --gpu-id 1
  #  done

  feat=fb40_wcmvn
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_wcmvn \
      --nj 14 \
      --model ${model} \
      --embedding-size 128 \
      --feat-dim 40 \
      --remove-vad \
      --extract \
      --valid \
      --resume Data/checkpoint/TDNN/fb40_wcmvn/soft_fix/checkpoint_40.pth \
      --xvector-dir Data/xvectors/TDNN/fb40_wcmvn/soft_fix \
      --loss-type soft \
      --num-valid 2 \
      --gpu-id 1
  done

fi

#stage=200
if [ $stage -le 20 ]; then
  model=LoResNet10
  feat=spect
  datasets=libri
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/libri/spect/dev_noc \
    #      --test-dir ${lstm_dir}/data/libri/spect/test_noc \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 4,32,128 \
    #      --embedding-size 128 \
    #      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}/checkpoint_15.pth \
    #      --loss-type soft \
    #      --dropout-p 0.25 \
    #      --num-valid 1 \
    #      --gpu-id 1
    #
    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/libri/spect/dev_noc \
    #      --test-dir ${lstm_dir}/data/libri/spect/test_noc \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 4,32,128 \
    #      --embedding-size 128 \
    #      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}_var/checkpoint_15.pth \
    #      --loss-type soft \
    #      --dropout-p 0.25 \
    #      --num-valid 1 \
    #      --gpu-id 1
    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/libri/spect/dev_noc \
    #      --test-dir ${lstm_dir}/data/libri/spect/test_noc \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 4,32,128 \
    #      --embedding-size 128 \
    #      --alpha 9.8 \
    #      --extract \
    #      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft/checkpoint_15.pth \
    #      --xvector-dir Data/xvectors/LoResNet10/libri/spect_noc/soft_128 \
    #      --loss-type ${loss} \
    #      --dropout-p 0.25 \
    #      --num-valid 2 \
    #      --gpu-id 1
    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/libri/spect/dev_noc \
      --test-dir ${lstm_dir}/data/libri/spect/test_noc \
      --nj 12 \
      --model ${model} \
      --channels 4,32,128 \
      --embedding-size 128 \
      --alpha 9.8 \
      --extract \
      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft_fix_43/checkpoint_15.pth \
      --xvector-dir Data/xvectors/LoResNet10/libri/spect_noc/soft_128 \
      --loss-type ${loss} \
      --dropout-p 0.25 \
      --num-valid 2 \
      --gpu-id 1

    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/libri/spect/dev_noc \
    #      --test-dir ${lstm_dir}/data/libri/spect/test_noc \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 4,16,64 \
    #      --embedding-size 128 \
    #      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft_var/checkpoint_15.pth \
    #      --loss-type soft \
    #      --dropout-p 0.25 \
    #      --num-valid 2 \
    #      --gpu-id 1
  done
fi

#stage=250
if [ $stage -le 25 ]; then
  model=LoResNet10
  feat=spect_wcmvn
  datasets=timit
  for loss in soft; do
    #    echo -e "\033[31m==> Loss type: ${loss} variance_fix length \033[0m"
    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/timit/spect/train_noc \
    #      --test-dir ${lstm_dir}/data/timit/spect/test_noc \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 4,16,64 \
    #      --embedding-size 128 \
    #      --resume Data/checkpoint/LoResNet10/timit_spect/soft_fix/checkpoint_15.pth \
    #      --loss-type soft \
    #      --dropout-p 0.25 \
    #      --num-valid 2 \
    #      --gpu-id 1

    echo -e "\033[31m==> Loss type: ${loss} variance_fix length \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/timit/spect/train_noc \
      --test-dir ${lstm_dir}/data/timit/spect/train_noc \
      --nj 12 \
      --model ${model} \
      --xvector-dir Data/xvectors/LoResNet10/timit_spect/soft_var \
      --channels 4,16,64 \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft_var/checkpoint_15.pth \
      --loss-type soft \
      --dropout-p 0.25 \
      --num-valid 2 \
      --gpu-id 1
  done
fi

if [ $stage -le 26 ]; then
  feat_type=spect
  feat=log
  loss=soft
  encod=None
  dataset=timit
  block_type=None

  for loss in soft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model LoResNet \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/train_${feat} \
      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/train_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${dataset}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${dataset}/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm None \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 128 \
      --loss-type ${loss} \
      --encoder-type None \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2 \
      --channels 4,16,64 \
      --alpha 10.8 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --test-input var \
      --frame-shift 300 \
      --dropout-p 0.5 \
      --xvector-dir Data/xvector/LoResNet8/timit/spect_egs_log/soft_dp05/epoch_12_var \
      --resume Data/checkpoint/LoResNet8/timit/spect_egs_log/soft_dp05/checkpoint_12.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

# ===============================    ResNet20    ===============================

#stage=100
if [ $stage -le 30 ]; then
  model=ResNet20
  feat=spect_wcmvn
  datasets=vox
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} fix length \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_spect/dev_257_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_spect/test_257_wcmvn \
      --nj 12 \
      --model ${model} \
      --embedding-size 128 \
      --resume Data/checkpoint/ResNet20/spect_257_wcmvn/soft_dp0.5/checkpoint_24.pth \
      --loss-type soft \
      --dropout-p 0.5 \
      --num-valid 2 \
      --gpu-id 1
  done
fi

#stage=100
if [ $stage -le 40 ]; then
  model=ExResNet34
  #  for loss in soft asoft ; do
  for loss in soft; do
    echo -e "\n\033[1;4;31m Test ${model} with ${loss} vox_wcmvn\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb64_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb64_wcmvn \
      --nj 12 \
      --epochs 30 \
      --model ExResNet34 \
      --remove-vad \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --kernel-size 3,3 \
      --stride 1 \
      --time-dim 1 \
      --avg-size 1 \
      --resume Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_14/checkpoint_22.pth \
      --xvector-dir Data/xvectors/ExResNet34/vox1/fb64_wcmvn/soft_14 \
      --input-per-spks 192 \
      --num-valid 2 \
      --extract \
      --gpu-id 1 \
      --loss-type ${loss}

    #    echo -e "\n\033[1;4;31m Test ${model} with ${loss} vox_noc \033[0m\n"
    #    python -W ignore TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/Vox1_pyfb64/dev_noc \
    #      --test-dir ${lstm_dir}/data/Vox1_pyfb64/test_noc \
    #      --nj 12 \
    #      --epochs 30 \
    #      --model ExResNet34 \
    #      --remove-vad \
    #      --resnet-size 34 \
    #      --embedding-size 128 \
    #      --feat-dim 64 \
    #      --kernel-size 3,3 \
    #      --stride 1 \
    #      --avg-size 1 \
    #      --resume Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_14/checkpoint_22.pth \
    #      --input-per-spks 192 \
    #      --time-dim 1 \
    #      --extract \
    #      --num-valid 2 \
    #      --loss-type ${loss}
  done
fi

# ===============================    TDNN    ===============================

if [ $stage -le 50 ]; then
  #  for loss in soft asoft ; do
  model=SiResNet34
  datasets=vox1
  feat=fb64_mvnorm
  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb64 \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb64 \
      --nj 14 \
      --epochs 40 \
      --model ${model} \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --remove-vad \
      --extract \
      --valid \
      --kernel-size 3,3 \
      --stride 1 \
      --mvnorm \
      --test-input fix \
      --test-input-per-file 4 \
      --xvector-dir Data/xvectors/${model}/${datasets}/${feat}/${loss} \
      --resume Data/checkpoint/SiResNet34/vox1/fb64_cmvn/soft/checkpoint_21.pth \
      --input-per-spks 192 \
      --gpu-id 1 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi

# ===============================    TDNN    ===============================

if [ $stage -le 55 ]; then
  #  for loss in soft asoft ; do
  model=GradResNet
  datasets=vox1
  feat=fb64_mvnorm
  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/vox1/spect/dev_power \
      --test-dir ${lstm_dir}/data/vox1/spect/test_power \
      --nj 12 --epochs 18 \
      --model ${model} --resnet-size 8 \
      --inst-norm \
      --embedding-size 128 \
      --feat-dim 161 --valid \
      --test-input fix \
      --test-input-per-file 4 \
      --xvector-dir Data/xvector/GradResNet8_inst/vox1_power/spect_time/soft_dp25 \
      --resume Data/checkpoint/GradResNet8_inst/vox1_power/spect_time/soft_dp25/checkpoint_18.pth \
      --input-per-spks 224 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 56 ]; then
  #  for loss in soft asoft ; do
  model=GradResNet
  datasets=vox1
  feat=spect
  for loss in mulcenter center; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/vox1/spect/dev_power \
      --test-dir ${lstm_dir}/data/vox1/spect/test_power \
      --nj 12 \
      --epochs 18 \
      --model ${model} \
      --resnet-size 8 \
      --inst-norm \
      --embedding-size 128 \
      --feat-dim 161 \
      --valid \
      --test-input fix \
      --stride 2 \
      --xvector-dir Data/xvector/GradResNet8/vox1_power/spect_egs/${loss}_dp25 \
      --resume Data/checkpoint/GradResNet8/vox1/spect_egs/${loss}_dp25/checkpoint_24.pth \
      --input-per-spks 224 \
      --gpu-id 0 \
      --num-valid 2 \
      --extract \
      --loss-type ${loss}
  done
fi

# stage=100
if [ $stage -le 60 ]; then
  dataset=army
  resnet_size=10
  for loss in soft; do # 32,128,512; 8,32,128
    #  Data/xvector/LoResNet10/army_v1/spect_egs_mean/soft_dp01
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --model LoResNet \
      --train-dir ${lstm_dir}/data/${dataset}/spect/dev_8k \
      --test-dir ${lstm_dir}/data/${dataset}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-per-spks 224 \
      --nj 16 \
      --embedding-size 128 \
      --channels 64,128,256,256 \
      --loss-type ${loss} \
      --test-input fix \
      --time-dim 1 \
      --test-input-per-file 4 \
      --inst-norm \
      --stride 2 \
      --dropout-p 0.1 \
      --xvector-dir Data/xvector/LoResNet${resnet_size}/army_v1/spect_egs_mean/soft_dp01 \
      --resume Data/checkpoint/LoResNet${resnet_size}/army_v1/spect_egs_mean/soft_dp01/checkpoint_24.pth \
      --trials trials \
      --gpu-id 0
  done

fi

if [ $stage -le 74 ]; then
  feat_type=klfb
  feat=fb40
  loss=arcsoft
  model=TDNN_v5
  encod=STAP
  dataset=vox2
  subset=test
  test_set=vox1
  input_dim=40
  input_norm=Mean

  # Training set: voxceleb 2 40-dimensional log fbanks kaldi  Loss: arcsoft sgd exp
  # Cosine Similarity
  #
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #|     vox1-test     |   2.3913%   |   0.2701    |   0.2368    |    0.3318    | 20211115 09:10:35 |

  for embedding_size in 512; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_valid \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --channels 512,512,512,512,1500 \
      --stride 1,1,1,1 \
      --margin 0.2 --s 30 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/${model}/${dataset}/${feat_type}_egs_baseline/${loss}_sgd_exp/input${input_norm}_${encod}_em${embedding_size}_wde4_var/${test_set}_${subset}_epoch_50_var \
      --resume Data/checkpoint/${model}/${dataset}/${feat_type}_egs_baseline/${loss}_sgd_exp/input${input_norm}_${encod}_em${embedding_size}_wde4_var/checkpoint_50.pth \
      --gpu-id 1 \
      --remove-vad \
      --cos-sim
  done
  exit
fi

if [ $stage -le 75 ]; then
  feat_type=klfb feat=fb40
  loss=arcsoft
  model=TDNN_v5
  encod=STAP embedding_size=512
  dataset=vox1 test_set=vox1 subset=test
  input_dim=40 input_norm=Mean

  # Training set: voxceleb 1 40-dimensional log fbanks kaldi  Loss: arcsoft
  # Cosine Similarity
  #
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #|     vox1-test     |   4.3054%   |   0.2299    |   0.4212    |    0.6275    | 20211104 15:29:25 | embedding_size=256
  #|     vox1-test     |   4.3478%   |   0.1949    |   0.4755    |    0.5709    | 20211104 15:30:08 | embedding_size=512

  # Training set: voxceleb 1 40-dimensional log fbanks kaldi  Loss: arcsoft  sgd exp
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #|     vox1-test     |   4.9841%   |   0.1783    |   0.5531    |    0.6566    | 20211115 09:03:38 |

  # Training set: voxceleb 1 40-dimensional log fbanks kaldi epoch 45 where lr=0.0001
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #|     vox1-test     |   4.4115%   |   0.1958    |   0.4680    |    0.5846    | 20211130 17:14:14 | arcsoft
  #|     vox1-test     |   4.7561%   |   0.1878    |   0.4709    |    0.6445    | 20211130 17:18:04 | minarcsoft*0.1+arc
  #|     vox1-dev      |   0.1744%   |   0.3415    |   0.0157    |    0.0271    | 20211130 17:53:52 | arcsoft
  #|     vox1-dev      |   0.1552%   |   0.3450    |   0.0147    |    0.0334    | 20211130 17:35:29 | minarcsoft*0.1+arc
  # 20211130 18:08 min seems to be better on dev set
  echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
  for lamda_beta in 0.2 0.5 1.0 2; do # 32,128,512; 8,32,128
#    model_dir=TDNN_v5/vox1/klfb_egs_baseline/arcsoft/featfb40_inputMean_STAP_em512_wd5e4_var
    model_dir=TDNN_v5/vox1/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_STAP_em512_wd5e4_vares_dist_mani-1_lamda${lamda_beta}/123456

    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_v2 \
      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_valid_v2 \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi --nj 8 \
      --input-norm ${input_norm} --input-dim ${input_dim} --remove-vad \
      --encoder-type ${encod} --embedding-size ${embedding_size} \
      --channels 512,512,512,512,1500 --stride 1,1,1,1 \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --test-input fix --frame-shift 300 \
      --xvector-dir Data/xvector/${model_dir}/${test_set}_${subset}_best_var \
      --resume Data/checkpoint/${model_dir}/best.pth \
      --gpu-id 1 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 76 ]; then
  feat_type=pyfb feat=fb40
  loss=soft
  model=TDNN_v5
  encod=None
  dataset=vox1 subset=test test_set=sitw

  # Training set: voxceleb 1 40-dimensional log fbanks ws25  Loss: soft
  # Cosine Similarity
  #
  # |   Test Set   |   EER ( % ) | Threshold | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  # +--------------+-------------+-----------+-------------+--------------+-------------------+
  # |  vox1-test   |   4.5864%   |   0.2424    |   0.4426    |    0.5638    | 20210531 17:00:32 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # | cnceleb-test |  16.6445%   |   0.2516    |   0.7963    |    0.9313    | 20210531 17:37:23 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # | aidata-test  |  10.8652%   |   0.3349    |   0.7937    |    0.9379    | 20210531 17:26:26 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |  magic-test  |  18.1604%   |   0.3161    |   0.9939    |    0.9977    | 20210531 17:13:37 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |   sitw dev      |    x%    |   0.2708  |    0.3919   |     0.5955   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+--------------+
  # |   sitw eval     |    x%    |   0.2695  |    0.4683   |     0.7143   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+-------------------+
  # |   magic-test    |   x%    |   0.3359  |    0.9984   |     0.9990   | 20210529 22:11:08 |
  # +-----------------+---------------+-----------+-------------+--------------+-------------------+
  # +-----------------+---------------+-----------+---------------+----------------+--------------+
  # |  aishell2 test  |   x%    |   0.2786811   |    0.8212      |     0.9527     |   20210515   |
  # +-----------------+---------------+-----------+---------------+----------------+--------------+
  # |   aidata-test   |    x%    |   0.3503  |    0.7233     |     0.9196     | 20210529 21:04:32 |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+

  # sitw dev Test  ERR: 4.0046%, Threshold: 0.2708 mindcf-0.01: 0.3919, mindcf-0.001: 0.5955.
  # Test  ERR: 4.5107%, Threshold: 0.2695 mindcf-0.01: 0.4683, mindcf-0.001: 0.7143

  for subset in dev eval; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat}_ws25 \
      --feat-format kaldi --nj 12 \
      --input-norm Mean --input-dim 40 \
      --embedding-size 256 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 --s 30 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4_var/${test_set}_${subset}_epoch_40_var \
      --resume Data/checkpoint/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4_var/checkpoint_40.pth \
      --gpu-id 1 --remove-vad \
      --cos-sim
  done
  exit
fi

if [ $stage -le 77 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  model=TDNN_v5
  encod=None
  dataset=aishell2
  test_set=aishell2

  # Training set: aishell 2 Loss: arcosft

  # |   Test Set      |   EER (%)  |   Threshold   |  MinDCF-0.01   |   MinDCF-0.01  |     Date     |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |  aishell2 test  |    1.4740%    |   0.2053137   |    0.2740      |     0.4685     |   20210517   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |   vox1 test     |   22.3118%    |   0.2578884   |    0.8733      |     0.8923     |   20210517   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |  aidata test    |   11.4180%    |   0.3180055   |    0.7140      |     0.8919     |   20210517   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |  cnceleb test   |   27.6964%    |   0.2075080   |    0.9081      |     0.9997     |   20210517   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/aishell2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/aishell2/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/aishell2/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --trials trials_30w \
      --feat-format kaldi \
      --input-norm Mean \
      --block-type basic \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --stride 1 \
      --margin 0.25 \
      --s 30 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/${test_set}_${subset}_epoch_60_var \
      --resume Data/checkpoint/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 78 ]; then
  feat_type=klsp
  feat=log
  loss=arcsoft
  model=TDNN_v5
  encod=None
  dataset=vox1
  test_set=vox1

#  for subset in test; do # 32,128,512; 8,32,128
#    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset} \
#      --trials trials \
#      --feat-format kaldi \
#      --input-norm Mean \
#      --block-type basic \
#      --input-dim 161 \
#      --nj 12 \
#      --embedding-size 512 \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --channels 512,512,512,512,1500 \
#      --stride 1 \
#      --margin 0.2 \
#      --s 30 \
#      --test-input var \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/vox1/klsp_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/${test_set}_${subset}_epoch_40_var \
#      --resume Data/checkpoint/TDNN_v5/vox1/klsp_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/checkpoint_40.pth \
#      --gpu-id 0 \
#      --cos-sim
#  done

#  for subset in test; do # 32,128,512; 8,32,128
#    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset} \
#      --trials trials \
#      --feat-format kaldi \
#      --input-norm Mean \
#      --block-type basic \
#      --input-dim 161 \
#      --nj 12 \
#      --embedding-size 512 \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --channels 256,256,256,256,768 \
#      --stride 1 \
#      --margin 0.2 \
#      --s 30 \
#      --test-input var \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/vox1/klsp_egs_baseline/arcsoft/Mean_STAP_em512_chn256_wd5e4_var/${test_set}_${subset}_epoch_40_var \
#      --resume Data/checkpoint/TDNN_v5/vox1/klsp_egs_baseline/arcsoft/Mean_STAP_em512_chn256_wd5e4_var/checkpoint_40.pth \
#      --gpu-id 0 \
#      --cos-sim
#  done
  subset=test
  for weight in mel clean aug vox2; do
    mask_layer=attention
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev \
      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_valid \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset} \
      --trials trials \
      --feat-format kaldi \
      --input-norm Mean \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --block-type basic \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 256,256,256,256,768 \
      --stride 1 \
      --margin 0.2 \
      --s 30 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox1/klsp_egs_attention/arcsoft/Mean_STAP_em512_${weight}_wd5e4_var/${test_set}_${subset}_epoch_40_var \
      --resume Data/checkpoint/TDNN_v5/vox1/klsp_egs_attention/arcsoft/Mean_STAP_em512_${weight}_wd5e4_var/checkpoint_40.pth \
      --gpu-id 0 \
      --cos-sim
  done

  exit
fi

if [ $stage -le 79 ]; then
  feat_type=pyfb
  feat=fb40
  loss=arcsoft
  model=TDNN_v5
  encod=None
  dataset=vox2
  test_set=cnceleb

  # Training set: voxceleb 2 40-dimensional log fbanks ws25  Loss: soft
  # Cosine Similarity
  # Data/checkpoint/TDNN_v5/vox2/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em512_wd5e4_var/checkpoint_40.pth
  # |   Test Set      |    EER ( % )  | Threshold | MinDCF-0.01 | MinDCF-0.001 |     Date     |
  # +-----------------+---------------+-----------+-------------+--------------+--------------+
  # |   vox1 test     |    2.6670%    |   0.2869  |    0.2984   |     0.4581   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+--------------+
  # |  cnceleb test   |   13.8038%    |   0.2597  |    0.7632   |     0.9349   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+--------------+
  # |   sitw dev      |    4.0046%    |   0.2708  |    0.3919   |     0.5955   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+--------------+
  # |   sitw eval     |    4.5107%    |   0.2695  |    0.4683   |     0.7143   | 20210529     |
  # +-----------------+---------------+-----------+-------------+--------------+-------------------+
  # |   magic-test    |   14.8940%    |   0.3359  |    0.9984   |     0.9990   | 20210529 22:11:08 |
  # +-----------------+---------------+-----------+-------------+--------------+-------------------+
  # +-----------------+---------------+-----------+---------------+----------------+--------------+
  # |  aishell2 test  |   10.8300%    |   0.2787  |    0.8212      |     0.9527     |   20210515   |
  # +-----------------+---------------+-----------+---------------+----------------+--------------+
  # |   aidata-test   |    8.7480%    |   0.3503  |    0.7233     |     0.9196     | 20210529 21:04:32 |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+

  # sitw dev Test  ERR: 4.0046%, Threshold: 0.2708 mindcf-0.01: 0.3919, mindcf-0.001: 0.5955.
  # Test  ERR: 4.5107%, Threshold: 0.2695 mindcf-0.01: 0.4683, mindcf-0.001: 0.7143

  # Data/checkpoint/TDNN_v5/vox2/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em512_wde4_var/checkpoint_50.phi_theta
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |  vox1-test   |   2.3277%   |   0.3319    |   0.2805    |    0.4108    | 20210812 20:38:25 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # | cnceleb-test |  15.5626%   |   0.2350    |   0.7575    |    0.8728    | 20210812 21:10:24 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |   sitw-dev   |   2.8109%   |   0.3261    |   0.3045    |    0.4622    | 20210813 13:43:48 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  # |  sitw-eval   |   3.4445%   |   0.3201    |   0.3290    |    0.5059    | 20210813 13:45:34 |
  # +--------------+-------------+-------------+-------------+--------------+-------------------+
  #  for test_set in vox1 aishell2; do # 32,128,512; 8,32,128
  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
    subset=test
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --trials trials_${s} \
      --score-suffix ${s} \
      --valid-dir ${lstm_dir}/data/vox2/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat}_ws25 \
      --feat-format kaldi \
      --input-norm Mean \
      --block-type basic \
      --input-dim 40 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --stride 1,1,1,1 \
      --margin 0.3 \
      --s 15 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em512_wde4_var/${test_set}_${subset}_epoch_50_var \
      --resume Data/checkpoint/TDNN_v5/vox2/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em512_wde4_var/checkpoint_50.pth \
      --gpu-id 0 \
      --remove-vad \
      --cos-sim
  done
  exit
fi

if [ $stage -le 80 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  model=TDNN_v5
  encod=None
  dataset=vox1
  test_set=aidata

  # Training set: voxceleb 2 161-dimensional spectrogram  Loss: arcosft
  # Cosine Similarity
  #
  # |   Test Set      |    EER ( % )  |   Threshold   |  MinDCF-0.01   |   MinDCF-0.01  |     Date     |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |   vox1 test     |    2.3542%    |   0.2698025   |    0.2192      |     0.2854     |   20210426   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |   sitw dev      |    2.8109%    |   0.2630014   |    0.2466      |     0.4026     |   20210515   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |   sitw eval     |    3.2531%    |   0.2642460   |    0.2984      |     0.4581     |   20210515   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |  cnceleb test   |   16.8276%    |   0.2165570   |    0.6923      |     0.8009     |   20210515   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |  aishell2 test  |   10.8300%    |   0.2786811   |    0.8212      |     0.9527     |   20210515   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+
  # |   aidata test   |   10.0972%    |   0.2952531   |    0.7859      |     0.9520     |   20210515   |
  # +-----------------+---------------+---------------+----------------+----------------+--------------+

  # 20210515
  # test_set=sitw
  # dev
  # Test ERR is 2.8109%, Threshold is 0.2630014419555664
  #  mindcf-0.01 0.2466, mindcf-0.001 0.4026.
  # eval
  # Test ERR is 3.2531%, Threshold is 0.26424601674079895
  #  mindcf-0.01 0.2984, mindcf-0.001 0.4581.

  # 20210515
  # test_set=cnceleb
  # aishell2 test 30w trials
  # aidata test 50w trials

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
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
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/${test_set}_${subset}_epoch_60_var \
      --resume Data/checkpoint/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

# ===============================    RET    ===============================
if [ $stage -le 81 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  encod=None
  dataset=vox1
  block_type=Basic
  model=RET
  for loss in arcsoft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model RET \
      --train-dir ${lstm_dir}/data/vox2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --channels 512,512,512,512,512,1500 \
      --nj 12 \
      --alpha 0 \
      --margin 0.25 \
      --s 30 \
      --block-type ${block_type} \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/checkpoint/RET/vox2/spect_STAP_v2/arcsoft_100ce/emsize512_inputMean_Basic/epoch_25_var \
      --resume Data/checkpoint/RET/vox2/spect_STAP_v2/arcsoft_100ce/emsize512_inputMean_Basic/checkpoint_25.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi

if [ $stage -le 82 ]; then
  feat_type=spect
  feat=log
  input_norm=Mean
  loss=arcsoft
  encod=STAP
  dataset=vox1
  block_type=cbam
  model=RET
  embedding_size=512
  train_set=vox2
  # test_set=vox1
  # 1.6172%, Threshold is 0.29920902848243713
  # mindcf-0.01 0.1592, mindcf-0.001 0.2065.

  # test_set=sitw
  # dev
  # Test ERR is 2.3489%, Threshold is 0.2773
  # mindcf-0.01 0.2098, mindcf-0.001 0.3596.
  # eval
  # Test ERR is 2.6791%, Threshold is 0.2732
  # mindcf-0.01 0.2346, mindcf-0.001 0.4054.

  test_set=cnceleb
  # 20210515
  #  Test ERR is 19.5295%, Threshold is 0.28571683168411255
  #  mindcf-0.01 0.7313, mindcf-0.001 0.8193.

  for loss in arcsoft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 14 \
      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --input-dim 161 \
      --channels 512,512,512,512,512,1536 \
      --context 5,5,5 \
      --nj 12 \
      --alpha 0 \
      --margin 0.25 \
      --s 30 \
      --block-type ${block_type} \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --test-input var \
      --xvector-dir Data/xvector/RET14/vox2/spect_STAP_v2/arcsoft_0ce/em512_inputMean_cbam_bs128_wde4_shuf/${test_set}_test_epoch20_var \
      --resume Data/checkpoint/RET14/vox2/spect_STAP_v2/arcsoft_0ce/em512_inputMean_cbam_bs128_wde4_shuf/checkpoint_20.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 83 ]; then
  feat_type=pyfb
  feat=fb40_ws25
  input_norm=Mean
  loss=soft
  encod=STAP
  block_type=basic
  model=TDNN_v5
  embedding_size=256
  train_set=cnceleb
  test_set=cnceleb
  # 20210515
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #|   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  16.8387%   |   0.1933    |   0.7987    |    0.8964    | 20210825 20:54:13 |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-spee |   7.9099%   |   0.2843    |   0.4350    |    0.5942    | 20210825 21:01:33 |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-sing |  25.5825%   |   0.1310    |   0.9821    |    0.9965    | 20210825 21:06:39 |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+

  # 20210902 test with fix length 300 frames
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #|   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  14.9190%   |   0.1512    |   0.7366    |    0.8458    | 20210902 12:45:19 | soft
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  14.8080%   |   0.1580    |   0.7482    |    0.8556    | 20210902 12:50:19 | arcsoft
  #+--------------+-------------+-------------+-------------+--------------+-------------------+

  # 20210902 test with fix length 300 frames arcsoft
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-adve |  26.3158%   |   0.1290    |   1.0000    |    1.0000    | 20210902 15:41:44 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-dram |  17.0068%   |   0.1298    |   0.8836    |    0.9730    | 20210902 15:41:51 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-ente |  16.8028%   |   0.1429    |   0.8303    |    0.9222    | 20210902 15:42:13 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-inte |  12.7590%   |   0.1756    |   0.7412    |    0.8366    | 20210902 15:42:59 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-live |  10.3733%   |   0.1970    |   0.5995    |    0.7959    | 20210902 15:43:14 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-movi |  19.7368%   |   0.1273    |   0.9162    |    0.9430    | 20210902 15:43:18 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-play |  14.0000%   |   0.1541    |   0.8593    |    0.9600    | 20210902 15:43:22 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-reci |   8.5714%   |   0.2315    |   0.6518    |    0.8383    | 20210902 15:43:26 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-sing |  25.4834%   |   0.0985    |   0.9856    |    0.9975    | 20210902 15:43:39 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-spee |   5.8965%   |   0.2486    |   0.3730    |    0.5044    | 20210902 15:43:52 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-vlog |  11.3402%   |   0.1905    |   0.7051    |    0.8299    | 20210902 15:43:58 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+

  # 20211025 test with fix length 300 frames soft
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|    cnceleb-dev    |   5.8668%   |   0.1983    |   0.5304    |    0.7234    | 20211026 14:29:48 |  30w trials
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|  cnceleb-dev-dev  |   5.7318%   |   0.1989    |   0.5328    |    0.8036    | 20211026 19:46:28 | 640w
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|    cnceleb-test   |  14.9190%   |   0.1512    |   0.7366    |    0.8458    | 20210902 12:45:19 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-adve |  26.3158%   |   0.0976    |   0.9997    |    1.0000    | 20211025 22:09:29 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-dram |  15.1927%   |   0.1357    |   0.8360    |    0.9366    | 20211025 22:09:41 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-ente |  15.7526%   |   0.1438    |   0.8186    |    0.9053    | 20211025 22:10:18 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-inte |  13.6238%   |   0.1683    |   0.7267    |    0.8177    | 20211025 22:11:35 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-live |  10.5035%   |   0.1903    |   0.5658    |    0.7634    | 20211025 22:12:00 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-movi |  24.1228%   |   0.1062    |   0.9538    |    0.9912    | 20211025 22:12:08 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-play |  16.0000%   |   0.1427    |   0.9299    |    0.9600    | 20211025 22:12:14 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-reci |   5.7143%   |   0.2620    |   0.5899    |    0.8766    | 20211025 22:12:21 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-sing |  24.9380%   |   0.0765    |   0.9824    |    0.9961    | 20211025 22:12:42 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-spee |   6.3279%   |   0.2391    |   0.3689    |    0.4642    | 20211025 22:13:05 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-vlog |  10.8247%   |   0.1890    |   0.6853    |    0.7926    | 20211025 22:13:16 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+

  # 20211025 test with fix length 300 frames gradient reverse layers soft
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|   cnceleb-dev     |   3.8844%   |   0.2333    |   0.2444    |    0.3655    | 20211026 14:19:11 | 30w trials
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #|   cnceleb-test    |  15.5182%   |   0.1565    |   0.7982    |    0.8954    | 20211025 22:37:18 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-adve |  26.3158%   |   0.1151    |   1.0000    |    1.0000    | 20211025 22:37:26 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-dram |  17.3469%   |   0.1265    |   0.9295    |    0.9955    | 20211025 22:37:37 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-ente |  16.6861%   |   0.1481    |   0.8578    |    0.9481    | 20211025 22:38:13 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-inte |  13.9664%   |   0.1733    |   0.7933    |    0.8799    | 20211025 22:39:29 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-live |  10.9809%   |   0.1977    |   0.7429    |    0.9281    | 20211025 22:39:53 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-movi |  23.6842%   |   0.1128    |   0.9276    |    0.9605    | 20211025 22:40:01 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-play |  12.0000%   |   0.1706    |   0.9897    |    1.0000    | 20211025 22:40:07 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-reci |   8.5714%   |   0.2392    |   0.5378    |    0.7623    | 20211025 22:40:13 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-sing |  25.2851%   |   0.0965    |   0.9918    |    0.9980    | 20211025 22:40:35 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-spee |   6.4238%   |   0.2436    |   0.4618    |    0.6023    | 20211025 22:40:57 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test-vlog |  14.0464%   |   0.1789    |   0.6943    |    0.8328    | 20211025 22:41:08 |
  #+-------------------+-------------+-------------+-------------+--------------+-------------------+


  #  for loss in soft arcsoft; do # 32,128,512; 8,32,128
#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
#    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} \
#      --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials trials_${s} \
#      --score-suffix ${s} \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/test_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 \
#      --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/${test_set}_test_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/checkpoint_60.pth \
#      --gpu-id 0 \
#      --extract \
#      --cos-sim
#  done
#  for s in vlog; do
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} \
#      --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials subtrials/trials_vlog_${s} \
#      --score-suffix vl${s} \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 \
#      --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/checkpoint_60.pth \
#      --gpu-id 0 \
#      --extract \
#      --cos-sim
#  done

#  python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} \
#      --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials trials_640w \
#      --score-suffix dev_640w \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 \
#      --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
#      --gpu-id 1 \
#      --verbose 2 \
#      --cos-sim

#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
  for s in vlog; do
#    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 14 \
      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --trials subtrials/trials_vlog_${s} \
      --score-suffix vl${s} \
      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --input-dim 40 \
      --channels 512,512,512,512,1500 \
      --context 5,3,3,5 \
      --nj 12 \
      --alpha 0 \
      --margin 0.15 \
      --s 30 \
      --stride 1 \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type STAP \
      --test-input fix \
      --remove-vad \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
      --gpu-id 1 \
      --extract \
      --cos-sim
  done
  exit
#  for s in vlog; do
#    wc -l data/cnceleb/dev/subtrials/trials_vlog_${s}
#  done
fi

if [ $stage -le 84 ]; then
  feat_type=pyfb
  feat=fb40_ws25
  input_norm=Mean
  loss=arcsoft
  encod=STAP
  block_type=basic
  model=TDNN_v5
  embedding_size=256
  train_set=cnceleb
  test_set=cnceleb
  # 20210515
#      --trials trials_${s} \
#      --score-suffix ${s} \

  #  for loss in soft arcsoft; do # 32,128,512; 8,32,128
  for s in all; do
    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 14 \
      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --input-dim 40 \
      --channels 512,512,512,512,1500 \
      --context 5,3,3,5 \
      --nj 12 \
      --alpha 0 \
      --stride 1 \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type STAP \
      --test-input fix \
      --remove-vad \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/soft/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2dr1/${test_set}_test_epoch60_fix \
      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/soft/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2dr1/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

#exit
if [ $stage -le 90 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  encod=None
  dataset=vox1
  block_type=cbam
  for loss in arcsoft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model LoResNet \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/vox2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/aidata/${feat_type}/dev_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 256 \
      --loss-type ${loss} \
      --encoder-type None \
      --block-type ${block_type} \
      --kernel-size 5,7 \
      --stride 2,3 \
      --channels 64,128,256 \
      --alpha 0 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --test-input var \
      --dropout-p 0.5 \
      --xvector-dir Data/xvector/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57/epoch_40_var_aidata \
      --resume Data/checkpoint/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57/checkpoint_40.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 91 ]; then
  feat_type=spect
  model=LoResNet
  feat=log
  loss=soft
  encod=None
  dataset=vox1
  block_type=basic
  embedding_size=128

  for loss in soft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels 64,128,256 \
      --alpha 0 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --test-input var \
      --dropout-p 0.25 \
      --xvector-dir Data/xvector/LoResNet8/vox1/spect_egs_None/soft_dp25/epoch_20_var \
      --resume Data/checkpoint/LoResNet8/vox1/spect_egs_None/soft_dp25/checkpoint_20.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 92 ]; then
  feat_type=klsp
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=None
  alpha=0
  datasets=vox1
  testset=sitw
#  test_subset=
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8
#  sname=dev #dev_aug_com
  sname=dev_aug_com

  for test_subset in dev test; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels 64,128,256 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --test-input var \
      --dropout-p 0.25 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp25_wd5e4_${sname}_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp25_wd5e4_${sname}_var/checkpoint_40.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#|   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#|  vox1-test   |   3.2715%   |   0.2473    |   0.3078    |    0.4189    | 20210818 19:07:02 |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#| vox1-test-aug|   2.8367%   |   0.2615    |   0.2735    |    0.4051    | 20210818 19:11:29 |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
fi

if [ $stage -le 93 ]; then
  feat_type=klsp
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=SAP2
  alpha=0
  datasets=vox2
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8

  scheduler=rop
  optimizer=sgd
  input_norm=Mean
  chn=32
  sname=dev


  for chn in 32 16 ; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    if [ $chn -eq 64 ];then
      channels=64,128,256
      dp=0.25
      dp_str=25
    elif [ $chn -eq 32 ];then
      channels=32,64,128
      dp=0.2
      dp_str=20
    elif [ $chn -eq 16 ];then
      channels=16,32,64
      dp=0.125
      dp_str=125
    fi

    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp${dp_str}_alpha${alpha}_em${embedding_size}_chn${chn}_wde4_var
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --input-dim 161 \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels ${channels} \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --test-input var \
      --dropout-p ${dp} \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.2715%   |   0.2473    |   0.3078    |    0.4189    | 20210818 19:07:02 | chn64
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   2.8367%   |   0.2615    |   0.2735    |    0.4051    | 20210818 19:11:29 | chn64 vox1_aug
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   2.9852%   |   0.2544    |   0.3110    |    0.3925    | 20211129 14:03:34 | chn64 SAP2

# train vox2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   1.7285%   |   0.2826    |   0.1562    |    0.2301    | 20210917 16:54:39 | chn64
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   2.2694    |   0.2928    |   0.2025    |    0.2914    | 20220623 11:37:28 | chn32
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.5525    |   0.2793    |   0.3122    |    0.4336    | 20220623 11:38:25 | chn16


fi

if [ $stage -le 94 ]; then
  feat_type=klsp
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=None
  alpha=0
  datasets=vox1
  testset=sitw
#  test_subset=
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8
#  sname=dev #dev_aug_com
  sname=dev #_aug_com

  for test_subset in test; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels 32,64,128 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --test-input var \
      --dropout-p 0.2 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/Mean_cbam_None_dp20_alpha0_em256_wd5e4_chn32_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/Mean_cbam_None_dp20_alpha0_em256_wd5e4_chn32_var/checkpoint_50.pth \
      --gpu-id 0 \
      --cos-sim
  done
#  exit
fi

#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.6585%   |   0.2510    |   0.3411    |    0.4408    | 20210930 11:08:10 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

if [ $stage -le 95 ]; then
  feat_type=klsp
  model=LoResNet resnet_size=8
  feat=log
  loss=arcsoft
  encod=None
  alpha=0
  datasets=vox1 testset=vox1
#  test_subset=
  block_type=cbam
  encoder_type=AVG embedding_size=256
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  test_subset=test
  input_norm=Mean
  mask_layer=baseline
#  mask_layer=attention0
  kd_type=embedding_mse #vanilla #em_l2 vanilla
#  kd_loss=kld
  weight=clean
  scheduler=rop optimizer=sgd

  chn=16
  weight_norm=sum
  kd_ratio=0.4
  kd_loss=

  for chn in 16 ; do
  for seed in 123456 1234567 123458  ; do

#  for weight in clean vox2 ; do
#      for weight in mel clean aug vox2 ; do

    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    if [ $chn -eq 64 ];then
      channels=64,128,256
      dp=0.25
      dp_str=25
    elif [ $chn -eq 32 ];then
      channels=32,64,128
      dp=0.2
      dp_str=20
    elif [ $chn -eq 16 ];then
      channels=16,32,64
      dp=0.125
      dp_str=125
    fi

    if [[ $mask_layer == attention* ]];then
        at_str=_${weight}
        if [[ $weight_norm != max ]];then
          at_str=${at_str}${weight_norm}
        fi
    elif [ "$mask_layer" = "drop" ];then
        at_str=_${weight}_dp${weight_p}s${scale}
    else
      at_str=
    fi

#    check_path=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp${dp_str}_alpha${alpha}_em${embedding_size}${at_str}_chn${chn}_wd5e4_var
    check_path=${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp${dp_str}_alpha${alpha}_em${embedding_size}_wd5e4_chn${chn}_var_${kd_type}${kd_ratio}${kd_loss}
#    Data/checkpoint/LoResNet8/vox1/klsp_egs_kd_baseline/123458/arcsoft_sgd_rop/Mean_cbam_AVG_dp125_alpha0_em256_wd5e4_chn16_var_embedding_cos0.4
#    ${seed}
#    check_path=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${seed}/${loss}_sgd_rop/${input_norm}_${block_type}_${encoder_type}_dp${dp_str}_alpha${alpha}_em${embedding_size}_${weight}_chn${chn}_wd5e4_var
#    check_path=LoResNet8/vox1/klsp_egs_kd_baseline/${seed}/arcsoft_sgd_rop/Mean_cbam_AVG_dp${dp_str}_alpha0_em256_wd5e4_chn${chn}_var_em_cosmse
#    check_path=LoResNet8/vox1/klsp_egs_kd_baseline/arcsoft_sgd_rop/Mean_cbam_AVG_dp${dp_str}_alpha0_em256_wd5e4_chn${chn}_var_em_cosmse

    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-extract-dir ${lstm_dir}/data/${datasets}/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --weight-norm ${weight_norm} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels ${channels} \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --test-input var \
      --dropout-p ${dp} \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${check_path} \
      --resume Data/checkpoint/${check_path}/checkpoint_50.pth \
      --model-yaml Data/checkpoint/${check_path}/model.2022.07.01.yaml \
      --gpu-id 1 \
      --verbose 0 \
      --cos-sim
  done
  done
  exit
fi
#+-------------------+-------------+-------------+---------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  |  MinDCF-0.01  | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+---------------+--------------+-------------------+

#|     vox1-test     |  4.19±0.08  |             | 0.3688±0.0158 | 0.5157±0.0103 |                    chn16 baseline

#|   vox1-test-mel   |  4.27±0.07  |             | 0.3742±0.0120 | 0.4820±0.0467 |
#|  vox1-test-clean  |  4.09±0.02  |             | 0.3620±0.0045 | 0.5199±0.0123 |
#|  vox1-test-clean  |  4.15±0.06  |             | 0.3705±0.0068 | 0.4833±0.0054 | attention0 sum
#|   vox1-test-aug   |  4.15±0.09  |             | 0.3530±0.0207 | 0.5044±0.0375 |
#|  vox1-test-vox2   |  4.00±0.08  |             | 0.3698±0.0067 | 0.4834±0.0219 |

#|     vox1-test     |  4.13±0.00  |             | 0.3752±0.0167 | 0.4891±0.0306 | kd_emdeddings_cos


#|  vox1-test-rand   |   4.1994    |   0.2521    |   0.3782    |    0.5004    | 20220327 20:06:46 |
#|  vox1-test-randt  |   4.0668    |   0.2565    |   0.3700    |    0.5125    | 20220604 09:40:43 |

#|     vox1-test     |  3.51±0.14  |             | 0.3259±0.0120 | 0.4080±0.0359 | 20210930 11:08:10 |   chn32 baseline

#|   vox1-test-mel   |  3.26±0.12  |             | 0.3195±0.0081 | 0.4053±0.0321 |
#|  vox1-test-clean  |  3.38±0.08  |             | 0.3192±0.0160 | 0.3971±0.0204 |
#|   vox1-test-aug   |  3.33±0.04  |             | 0.3177±0.0018 | 0.4027±0.0188 |
#|  vox1-test-vox2   |  3.37±0.08  |             | 0.3002±0.0029 | 0.4035±0.0423 |

#|  vox1-test-randt  |   3.5154    |   0.2561    |   0.3427    |    0.4371    | 20220604 09:37:27 |

#|     vox1-test     |  3.28±0.02  |             | 0.2875±0.0144 | 0.3951±0.0311 | 20220814 20:04:36 |   chn64 baseline


if [ $stage -le 96 ]; then
  feat_type=klsp
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=AVG
  alpha=0
  datasets=vox1
  testset=vox1
#  test_subset=
  block_type=cbam
  encoder_type=AVG
  embedding_size=256
  resnet_size=8
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  test_subset=test
  input_norm=Mean
  mask_layer=baseline


  for weight in vox2 ; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"

    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_sgd_rop/${input_norm}_${block_type}_${encod}_dp20_alpha${alpha}_em${embedding_size}_wd5e4_chn32_var_vanillakld
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
      --feat-format kaldi \
      --input-norm Mean --input-dim 161 \
      --mask-layer ${mask_layer} --init-weight ${weight} \
      --nj 8 \
      --embedding-size ${embedding_size} \
      --loss-type ${loss} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 --stride 2,2 \
      --channels 32,64,128 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 \
      --test-input var \
      --dropout-p 0.12 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --gpu-id 1 \
      --cos-sim

#    _${weight}
  done
  exit
fi

#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# chn16
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.2100%   |   0.2511    |   0.3781    |    0.5214    | 20211004 15:15:00 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|  vox1-test-clea   |   4.1092%   |   0.2534    |   0.3652    |    0.5319    | 20211003 17:52:05 | attention vox1-clean
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.1676    |   0.2556    |   0.3846    |    0.4620    | 20220113 20:28:41 | kd soft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.1304    |   0.2488    |   0.3850    |    0.4499    | 20220228 18:25:20 | kd soft + emdeddings_l2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.1092    |   0.2500    |   0.3780    |    0.5180    | 20220807 20:08:10 | kd soft + emdeddings_cos
#|     vox1-test     |   4.2683    |   0.2433    |   0.3793    |    0.5153    | 20220807 20:01:01 |
#|     vox1-test     |   4.1888    |   0.2471    |   0.3434    |    0.5077    | 20220807 20:02:08 |
#|     vox1-test     |  4.19±0.06  |             | 0.3669±0.0166 | 0.5137±0.0044 |

#|     vox1-test     |  4.14±0.04  |             | 0.3688±0.0094 | 0.4828±0.0501 | kd soft kld
#|     vox1-test     |  4.13±0.00  |             | 0.3752±0.0167 | 0.4891±0.0306 | emdeddings_cos



# chn32
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.6585%   |   0.2510    |   0.3411    |    0.4408    | 20210930 11:08:10 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|  vox1-test-clea   |   3.2715%   |   0.2566    |   0.3187    |    0.3876    | 20210930 11:27:25 | attention vox1-clean
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.6161    |   0.2559    |   0.3371    |    0.4334    | 20220113 20:31:33 | kd soft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.4358    |   0.2541    |   0.3047    |    0.4223    | 20220228 18:31:14 | kd soft + emdeddings_l2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.4305    |   0.2535    |   0.3184    |    0.4331    | 20220807 20:09:25 | kd soft + emdeddings_cos
#|     vox1-test     |   3.3404    |   0.2527    |   0.3193    |    0.3802    | 20220807 20:03:25 |
#|     vox1-test     |   3.4040    |   0.2566    |   0.3311    |    0.3775    | 20220807 20:04:41 |
#|     vox1-test     |  3.39±0.04  |             | 0.3229±0.0058 | 0.3969±0.0256 |

#|     vox1-test     |  3.46±0.06  |             | 0.3252±0.0301 | 0.4324±0.0251 | kd soft kld

#|     vox1-test     |   3.3775    |   0.2592    |   0.3047    |    0.4303    | 20220216 18:14:06 | kd soft attention vox2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+


# ===============================    MultiResNet    ===============================

if [ $stage -le 100 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=18
  #  loss=soft
  encod=None
  transform=None
  loss_ratio=0.01
  alpha=13
  for loss in soft; do
    echo -e "\n\033[1;4;31m Testing ${model}_${resnet_size} in army with ${loss} kernel 5,5 \033[0m\n"

    python TrainAndTest/test_egs_multi.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir-a ${lstm_dir}/data/${datasets}/spect/aishell2_dev_8k_v4 \
      --train-dir-b ${lstm_dir}/data/${datasets}/spect/vox_dev_8k_v4 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2/trials_dir \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v4 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_valid_8k_v4 \
      --test-dir ${lstm_dir}/data/magic/spect/test_8k \
      --feat-format kaldi \
      --input-norm Mean --input-dim 81 \
      --batch-size 128 \
      --nj 10 \
      --lr 0.1 \
      --mask-layer freq --mask-len 20 \
      --stride 1 --fast \
      --xvector-dir Data/xvector/MultiResNet18/army_x4/spect_egs_None/soft/dp25_b256_13_fast_None_mask/epoch_36_var_magic \
      --resume Data/checkpoint/MultiResNet18/army_x4/spect_egs_None/soft/dp25_b256_13_fast_None_mask/checkpoint_36.pth \
      --channels 32,64,128,256 \
      --transform ${transform} \
      --encoder-type ${encod} --embedding-size 128 \
      --time-dim 1 --avg-size 4 \
      --num-valid 4 \
      --alpha ${alpha} \
      --margin 0.3 --s 30 --m 3 \
      --test-input var \
      --loss-ratio ${loss_ratio} \
      --weight-decay 0.0005 \
      --dropout-p 0.1 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
  exit
fi
#exit

if [ $stage -le 101 ]; then
  feat_type=spect feat=log
  loss=arcsoft
  encod=None
  dataset=army_v1
  block_type=None
  for loss in soft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model LoResNet --resnet-size 10 \
      --train-dir ${lstm_dir}/data/army/spect/dev_8k \
      --test-dir ${lstm_dir}/data/army/spect/test_8k \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 128 \
      --loss-type ${loss} \
      --encoder-type None \
      --block-type ${block_type} \
      --kernel-size 5,5 --stride 2 \
      --channels 64,128,256,256 \
      --alpha 12 \
      --margin 0.3 --s 30 --m 3 \
      --test-input var \
      --frame-shift 300 \
      --dropout-p 0.1 \
      --xvector-dir Data/xvector/LoResNet10/army_v1/spect_egs_mean/soft_dp01/epoch_20_var \
      --resume Data/checkpoint/LoResNet10/army_v1/spect_egs_mean/soft_dp01/checkpoint_20.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi


if [ $stage -le 200 ]; then
  model=ThinResNet resnet_size=18
  feat_type=klsp feat=log
  loss=arcsoft
  alpha=0
  datasets=vox1 testset=cnceleb
#  test_subset=
  block_type=basic downsample=k1
  encoder_type=SAP2 embedding_size=256
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  weight=v2_eer #v2_rclean_gax #mel
  weight_norm=norm scale=0.2 weight_p=0.1
  # v2_eer_norm_scale0.5
  echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"

  for mask_layer in drop; do
  for testset in vox1 aishell2 ;do
    for weight in v2_rclean_gean ; do
    for seed in 123456 123457 123458;do
      for test_subset in test ; do
        at_str=
        if [[ $mask_layer == attention* ]];then
          at_str=_${weight}_${weight_norm} #_mel_mean
        elif [[ $mask_layer == drop ]];then
          at_str=_${weight}_${weight_norm} #_scale${scale} #_mel_mean
        fi
        # Mean_batch128_basic_downk1_avg4_SAP2_em256_dp01_alpha0_none1_mel_max_scale0.2_wde4_varesmix2_bashuf2_dist
        # _<init_weight>_<weight_norm>_scale<scale>
        model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/arcsoft_sgd_rop/Mean_batch128_basic_downk1_avg4_SAP2_em256_dp01_alpha0_none1${at_str}_wde4_varesmix2_bashuf2_dist/${seed}
#        if [ $seed -eq 123456 ]; then
#          yaml_file=model.2022.11.09.yaml
#        elif [ $seed -eq 123457 ]; then
#          yaml_file=model.2022.11.10.yaml
#        else
#          yaml_file=model.2022.11.11.yaml
#        fi
        python -W ignore TrainAndTest/test_egs.py \
          --model ${model} --resnet-size ${resnet_size} \
          --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
          --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir --train-trials trials_2w \
          --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
          --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
          --feat-format kaldi --nj 8 \
          --input-norm Mean --input-dim 161 \
          --mask-layer ${mask_layer} --init-weight ${weight} --weight-norm ${weight_norm} --scale ${scale} --weight-p ${weight_p} \
          --kernel-size 5,5 --stride 2,2 --fast none1 \
          --block-type ${block_type} --downsample ${downsample} \
          --channels 16,32,64,128 \
          --loss-type ${loss} --margin 0.2 --s 30 \
          --time-dim 1 --avg-size 4 \
          --encoder-type ${encoder_type} --embedding-size ${embedding_size} --alpha ${alpha} \
          --test-input var --dropout-p 0.1 \
          --xvector-dir Data/xvector/${model_dir}/${testset}_${test_subset}_best_var \
          --resume Data/checkpoint/${model_dir}/best.pth \
          --check-yaml Data/checkpoint/${model_dir}/model.2022.07.01.yaml \
          --gpu-id 0 --verbose 0 \
          --cos-sim
      done
    done
    done
  done
  done
  exit
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
# ThinResNet18 klfb40
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-dev      |   1.0300    |   0.2914    |   0.1291    |    0.2308    | 20211221 22:57:16 |
#|     vox1-test     |   4.2683%   |   0.2369    |   0.3929    |    0.4767    | 20211119 15:09:05 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
# attention vox2_rcf
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-dev      |   0.9700    |   0.2949    |   0.1282    |    0.2440    | 20211221 23:07:13 |
#|     vox1-test     |   4.2895    |   0.2371    |   0.4231    |    0.5714    | 20211221 22:43:00 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# ThinResNet34 klsp
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-dev      |   0.4700    |   0.3264    |   0.0629    |    0.1140    | 20211221 23:30:06 |
#|     vox1-test     |   3.7116    |   0.2465    |   0.3838    |    0.4681    | 20211221 23:15:57 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
# attention vox2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-dev      |   0.4600    |   0.3283    |   0.0631    |    0.1092    | 20211221 23:43:06 |
#|     vox1-test     |   3.5790    |   0.2471    |   0.4012    |    0.5222    | 20211221 23:17:22 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
fi


if [ $stage -le 201 ]; then
  feat_type=klsp
  model=ThinResNet resnet_size=10
  feat=log
  loss=arcsoft
  encod=SAP2 embedding_size=256
  alpha=0
  datasets=vox1 testset=vox1
#  test_subset=
  block_type=cbam_v2
#  encoder_type=None
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  downsample=k5
  test_subset=test

  for sname in dev dev_aug_com; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"

    model_dir=ThinResNet${resnet_size}/vox1/klsp_egs_rvec/arcsoft_sgd_rop/chn32_Mean_cbam_v2_downk5_SAP2_em256_dp01_alpha0_none1_wd5e4_var_${sname}
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
      --feat-format kaldi --nj 12 \
      --input-norm Mean --input-dim 161 \
      --embedding-size ${embedding_size} \
      --encoder-type ${encod} \
      --block-type ${block_type} --downsample ${downsample} \
      --kernel-size 5,5 --stride 2,2 --fast none1 \
      --channels 32,64,128,256 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --time-dim 1 --avg-size 5 \
      --test-input var \
      --dropout-p 0.1 \
      --xvector-dir Data/xvector/${model_dir}/${test_subset}_epoch_60_var \
      --resume Data/checkpoint/${model_dir}/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim

#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} --resnet-size ${resnet_size} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset} \
#      --feat-format kaldi \
#      --input-norm Mean --input-dim 161 \
#      --nj 12 \
#      --mask-layer attention --init-weight vox2 \
#      --embedding-size ${embedding_size} \
#      --fast none1 \
#      --downsample ${downsample} \
#      --encoder-type ${encod} \
#      --block-type ${block_type} \
#      --kernel-size 5,5 --stride 2,2 \
#      --channels 16,32,64,128 \
#      --alpha ${alpha} \
#      --loss-type ${loss} --margin 0.2 --s 30 \
#      --time-dim 1 --avg-size 5 \
#      --test-input var \
#      --dropout-p 0.1 \
#      --xvector-dir Data/xvector/ThinResNet${resnet_size}/vox1/klsp_egs_rvec_attention/arcsoft/inputMean_basic_v2_downk5_AVG_em256_dp125_alpha0_none1_vox2_wd5e4_var/${test_subset}_epoch_50_var \
#      --resume Data/checkpoint/ThinResNet${resnet_size}/vox1/klsp_egs_rvec_attention/arcsoft/inputMean_basic_v2_downk5_AVG_em256_dp125_alpha0_none1_vox2_wd5e4_var/checkpoint_50.pth \
#      --gpu-id 0 \
#      --cos-sim
  done
  exit

#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#ThinResNet cbam_v2 dowsample=k5 SAP2 embedding_size=256 dropout=0.1 fast=none1

# chn16 dev
#|     vox1-test     |   3.1442    |   0.2504    |   0.3185    |    0.3870    | 20220209 17:05:27 | 34
#|     vox1-test     |   3.2768    |   0.2429    |   0.3349    |    0.4189    | 20220209 17:24:12 | 18
#|     vox1-test     |   3.4305    |   0.2457    |   0.3474    |    0.4059    | 20220214 16:57:46 | 10

# chn16 dev_aug
#|     vox1-test     |   2.6723    |   0.2723    |   0.2847    |    0.4488    | 20220209 17:07:34 | 34
#|     vox1-test     |   3.0064    |   0.2658    |   0.2949    |    0.3840    | 20220214 16:54:25 | 18
#|     vox1-test     |   3.3351    |   0.2662    |   0.3374    |    0.4496    | 20220209 18:13:51 | 10

# chn32 dev
#|     vox1-test     |   3.0859    |   0.2410    |   0.2972    |    0.3885    | 20220209 17:01:16 | 34
#|     vox1-test     |   3.2768    |   0.2429    |   0.3349    |    0.4189    | 20220209 17:24:12 | 18
#|     vox1-test     |   3.4305    |   0.2457    |   0.3474    |    0.4059    | 20220214 16:57:46 | 10

# chn32 dev_aug
#|     vox1-test     |   ------    |   0.2410    |   0.2972    |    0.3885    | 20220209 17:01:16 | 34
#|     vox1-test     |   2.7413    |   0.2583    |   0.2781    |    0.3520    | 20220209 17:27:02 | 18
#|     vox1-test     |   3.0011    |   0.2647    |   0.3277    |    0.3613    | 20220214 17:00:17 | 10

fi


if [ $stage -le 202 ]; then
  feat_type=klfb feat=log
  model=ThinResNet
  loss=arcsoft
  scheduler=rop optimizer=sgd
  input_dim=40 input_norm=Mean

  encoder_type=ASTP2 embedding_size=256
  alpha=0
  datasets=vox1 testset=vox1
#  test_subset=
  block_type=seblock red_ratio=2
#  encoder_type=None
#  resnet_size=18 10
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  downsample=k3
  test_subset=test
  mask_layer=baseline
  dp=0.1
  avg_size=5
  fast=none1
  chn=32
  batch_size=128

  kd_type=attention #em_l2 vanilla
  kd_ratio=1000
  kd_loss=

  attention_type=both norm_type=feat
#        --downsample ${downsample} \
#      --trials trials_20w \
  for resnet_size in 34 ;do
        echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
  for attention_type in freq ; do
  for seed in 123456 123457 123458 ; do
  for testset in vox1 ; do
    if [ $resnet_size -le 34 ];then
      expansion=1
    else
      expansion=2
    fi

    if [ $expansion -eq 1 ]; then
      exp_str=
    else
      exp_str=_exp${expansion}
    fi

    if [ $chn -eq 16 ]; then
        channels=16,32,64,128
        chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    elif [ $chn -eq 64 ]; then
      channels=64,128,256,512
      chn_str=chn64_
    fi

    at_str=
    if [[ $mask_layer == attention* ]];then
      at_str=_${weight}
    elif [ "$mask_layer" = "drop" ];then
      at_str=_${weight}_dp${weight_p}s${scale}
    fi

    if [ $dp = 0.25 ];then
      dp_str=25
    elif [ $dp = 0.2 ];then
      dp_str=02
    elif [ $dp = 0.125 ];then
      dp_str=125
    elif [ $dp = 0.1 ];then
      dp_str=01
    fi

    if [[ $kd_type == attention ]];then
      kd_ratio=1000
      if [[ $norm_type == input ]]; then
        kd_str=_${kd_type}${kd_ratio}${kd_loss}
      else
#          kd_ratio=40
        kd_str=_${kd_type}${kd_ratio}${kd_loss}_${attention_type}_${norm_type}
      fi
    else
      kd_ratio=0.4
      kd_str=_${kd_type}${kd_ratio}${kd_loss}
    fi

    # model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_red${red_ratio}${exp_str}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wd5e4_var${kd_str}_bashuf/${seed}
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_red${red_ratio}${exp_str}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wd5e4_varesmix2_bashuf2_dist/${seed}
    

    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_fb40 \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb40/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_fb40 \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset}_fb40 --trials trials \
      --feat-format kaldi --nj 8 \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --mask-layer ${mask_layer} --init-weight vox2_rcf \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} --red-ratio ${red_ratio} --expansion ${expansion} \
      --embedding-size ${embedding_size} \
      --kernel-size 5,5 --stride 2,1 --channels ${channels} \
      --downsample ${downsample} --fast ${fast} \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --time-dim 1 --avg-size ${avg_size} \
      --dropout-p ${dp} \
      --xvector-dir Data/xvector/${model_dir}/${testset}_${test_subset}_var \
      --resume Data/checkpoint/${model_dir}/best.pth \
      --gpu-id 0 \
      --remove-vad --test-input var \
      --verbose 0 \
      --cos-sim
  done
  done
  done
  done
  exit
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# Arcsoft fb40 thin34_basic_none1s
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.0403%   |   0.2446    |   0.4034    |    0.5689    | 20211130 16:56:42 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.9343    |   0.2493    |   0.3624    |    0.5463    | 20211222 20:15:48 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
# Arcsoft fb40 thin34_cbam
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.0138%   |   0.2442    |   0.4186    |    0.5426    | 20211203 16:16:54 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# Arcsoft fb40 thin18_basic_v2_k5
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.1676%   |   0.2354    |   0.3765    |    0.5602    | 20211202 15:09:09 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# Arcsoft fb40 thin34_basic vox2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   1.8452    |   0.2847    |   0.1973    |    0.3190    | 20211210 15:47:27 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   14.0979   |   0.2445    |   0.6538    |    0.8029    | 20211210 16:10:01 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   aishell2-test   |   9.9600    |   0.3146    |   0.7508    |    0.8895    | 20211210 19:20:30 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
fi

if [ $stage -le 202 ]; then
  data_path=${lstm_dir}/data/vox1/dev
  xvector_dir=Data/xvector/ThinResNet34/vox2/klfb_egs_baseline/arcsoft_sgd_rop/chn32_Mean_basic_downNone_none1_SAP2_dp01_alpha0_em256_wde4_var
  train_feat_dir=${xvector_dir}/vox1_dev_xvector_var

  for test_set in vox1 cnceleb aishell2;do
    echo -e "\n\033[1;4;31m Testing in ${test_set}: \033[0m"
    
    test_feat_dir=${xvector_dir}/${test_set}_test_xvector_var
    trials=${lstm_dir}/data/${test_set}/test/trials
    ./Score/plda.sh $data_path $train_feat_dir $test_feat_dir $trials
  done

# PLDA ThinResNet34
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   1.903     |    -----    |   0.3710    |    0.6383    | 20211210 15:47:27 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|  cnceleb-test     |   17.05     |    -----    |   0.7488    |    0.8584    | 20211210 15:47:27 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|  aishell2-test    |   12.17     |    -----    |   0.8454    |    0.9474    | 20211210 15:47:27 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
  exit
fi

if [ $stage -le 203 ]; then
  feat_type=klfb
  model=ThinResNet resnet_size=10
  feat=log
  loss=arcsoft
  encod=SAP2 alpha=0
  datasets=vox1 testset=vox1
#  test_subset=
  block_type=cbam
  encoder_type=None embedding_size=256
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  downsample=k3
  test_subset=test
  mask_layer=baseline
#        --downsample ${downsample} \
#      --trials trials_20w \
#      --mask-layer attention \
#      --init-weight vox2_rcf \

  for input_dim in 24 40 64 80 ; do
    if [ $input_dim -eq 40 ];then
      valid_dir=valid_fb40
    else
      valid_dir=dev_fb${input_dim}_valid
    fi
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"

    model_dir=ThinResNet${resnet_size}/vox1/${feat_type}{input_dim}_egs_${mask_layer}/arcsoft_sgd_rop/Mean_batch256_cbam_downk3_none1_SAP2_dp01_alpha0_em256_wd5e4_var
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${valid_dir} \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset}_fb${input_dim} \
      --trials trials \
      --feat-format kaldi --nj 8 \
      --input-norm Mean --input-dim ${input_dim} \
      --encoder-type ${encod} --embedding-size ${embedding_size} \
      --block-type ${block_type} --downsample ${downsample} \
      --kernel-size 5,5 --stride 2 --fast none1 \
      --channels 16,32,64,128 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --time-dim 1 --avg-size 5 \
      --test-input var \
      --dropout-p 0.1 \
      --xvector-dir Data/xvector/${model_dir}/${testset}_${test_subset}_best_var \
      --resume Data/checkpoint/${model_dir}/best.pth \
      --gpu-id 0 --verbose 0 \
      --remove-vad \
      --cos-sim
  done
  exit

# Arcsoft

#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# fb24 thin_cbam_none1
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.5864    |   0.2296    |   0.4324    |    0.4911    | 20220223 15:33:06 |  resnet10
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.8091    |   0.2303    |   0.4577    |    0.5708    | 20220221 17:25:00 |  resnet18
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.9099    |   0.2204    |   0.4437    |    0.5615    | 20220221 17:16:25 |  resnet34
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# fb40 thin_cbam_none1
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   4.1304    |   0.2402    |   0.3849    |    0.5364    | 20220223 15:34:24 |  resnet10
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.9873    |   0.2385    |   0.3932    |    0.5748    | 20220221 17:26:24 |  resnet18
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.9555    |   0.2415    |   0.3951    |    0.4800    | 20220221 17:18:05 |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# fb64 thin_cbam_none1
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.7593    |   0.2490    |   0.3611    |    0.4481    | 20220223 15:35:45 |  resnet10
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.4571    |   0.2541    |   0.3878    |    0.4640    | 20220221 17:27:55 |  resnet18
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# fb80 thin_cbam_none1
#|     vox1-test     |   3.7805    |   0.2483    |   0.3600    |    0.5120    | 20220223 15:37:08 |  resnet10
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     vox1-test     |   3.3457    |   0.2577    |   0.3532    |    0.5051    | 20220221 17:29:30 |  resnet18
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

fi


if [ $stage -le 300 ]; then
  feat_type=klfb feat=fb40
  input_norm=Mean
  loss=arcsoft
  block_type=basic
  model=TDNN_v5
  encod=STAP embedding_size=512
  train_set=cnceleb test_set=cnceleb subset=dev
  # 20210902 test
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #|   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  14.9190%   |   0.1512    |   0.7366    |    0.8458    | 20210902 12:45:19 | soft
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  14.3642%   |   0.1225    |   0.7395    |    0.8437    | 20210902 12:50:19 | arcsoft
  #+--------------+-------------+-------------+-------------+--------------+-------------------+
  #| cnceleb-test |  14.4973%   |   0.1296    |   0.7453    |    0.8601    | 20210902 12:50:19 | arcdist
  #+--------------+-------------+-------------+-------------+--------------+-------------------+

  # for test_set in cnceleb; do # 32,128,512; 8,32,128
#    --trials trials_${s} \
#      --score-suffix ${s} \
#      --extract \

#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do

# --xvector-dir Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/${test_set}_${subset}_epoch50_fix \
  #      --resume Data/checkpoint/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/checkpoint_50.pth \
  #   echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
  #   python -W ignore TrainAndTest/test_egs.py \
  #     --model ${model} --resnet-size 14 \
  #     --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
  #     --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
  #     --train-trials trials_2w \
  #     --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
  #     --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
  #     --feat-format kaldi \
  #     --input-norm ${input_norm} \
  #     --input-dim 40 \
  #     --channels 512,512,512,512,1500 \
  #     --context 5,3,3,5 \
  #     --nj 10 \
  #     --alpha 0 \
  #     --margin 0.2 --s 30 \
  #     --stride 1 \
  #     --block-type ${block_type} \
  #     --embedding-size ${embedding_size} \
  #     --loss-type ${loss} \
  #     --encoder-type STAP \
  #     --test-input fix \
  #     --remove-vad \
  #     --resume Data/checkpoint/TDNN_v5/cnceleb/klfb_egs_baseline/arcdist/Mean_STAP_em512_lr1_wd5e4_var/checkpoint_50.pth \
  #     --xvector-dir Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcdist/Mean_STAP_em512_lr1_wd5e4_var/${test_set}_${subset}_epoch50_fix \
  #     --frame-shift 300 \
  #     --gpu-id 0 \
  #     --cos-sim
  # done
       # --verbose 2 \

 # for s in adve_adve adve_spee dram_reci ente_reci inte_sing live_vlog play_sing sing_vlog adve_dram adve_vlog dram_sing ente_sing inte_spee movi_movi play_spee spee_spee adve_ente dram_spee ente_spee inte_vlog movi_play play_vlog spee_vlog adve_inte dram_dram dram_vlog ente_vlog live_live movi_reci reci_reci vlog_vlog adve_live dram_ente ente_ente inte_inte live_movi movi_sing reci_sing adve_movi dram_inte ente_inte inte_live live_play movi_spee reci_spee adve_play dram_live ente_live inte_movi live_reci movi_vlog reci_vlog adve_reci dram_movi ente_movi inte_play live_sing play_play sing_sing adve_sing dram_play ente_play inte_reci live_spee play_reci sing_spee; do
 for s in adad adpl drdr drre enli envl insi lire more plsi revl vlvl addr adre dren drsi enmo inin insp lisi mosi plsp sisi aden adsi drin drsp enpl inli invl lisp mosp plvl sisp adin adsp drli drvl enre inmo lili livl movl rere sivl adli advl drmo enen ensi inpl limo momo plpl resi spsp admo drpl enin ensp inre lipl mopl plre resp spvl; do
   python -W ignore TrainAndTest/test_egs.py \
     --model ${model} --resnet-size 14 \
     --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
     --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --trials subtrials/trials_${s} \
     --score-suffix ${s} \
     --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
     --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
     --feat-format kaldi --nj 12 \
     --input-norm ${input_norm} \
     --input-dim 40 \
     --channels 512,512,512,512,1500 \
     --context 5,3,3,5 --stride 1 \
     --alpha 0 \
     --loss-type ${loss} --margin 0.2 --s 30 \
     --block-type ${block_type} \
     --embedding-size ${embedding_size} \
     --encoder-type STAP \
     --test-input fix \
     --remove-vad \
     --frame-shift 300 \
     --xvector-dir Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/${test_set}_dev_epoch50_fix \
     --resume Data/checkpoint/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/checkpoint_50.pth \
     --gpu-id 0 \
     --extract \
     --cos-sim
 done

#  python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials trials_640w \
#      --score-suffix dev_640w \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
#      --gpu-id 1 \
#      --verbose 2 \
#      --cos-sim

#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
#  for s in vlog; do
##    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials subtrials/trials_vlog_${s} \
#      --score-suffix vl${s} \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
#      --gpu-id 1 \
#      --extract \
#      --cos-sim
#  done
  exit
#  for s in vlog; do
#    wc -l data/cnceleb/dev/subtrials/trials_vlog_${s}
#  done
fi

if [ $stage -le 301 ]; then
  feat_type=klfb
  input_norm=Mean
  loss=arcsoft
  model=ThinResNet resnet_size=18
  embedding_size=256
  train_set=cnceleb
  test_set=cnceleb
  dev_subset=dev test_subset=dev

  encoder_type=SAP2
  embedding_size=512
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcdist
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  input_dim=40
  feat=fb${input_dim}

  # for test_set in cnceleb; do # 32,128,512; 8,32,128
#    --trials trials_${s} \
#      --score-suffix ${s} \
#      --extract \
#--score-norm s-norm \

#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
  for trials in trials_500w ; do
# --xvector-dir Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/${test_set}_${subset}_epoch50_fix \
  #      --resume Data/checkpoint/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/checkpoint_50.pth \
     echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
     python -W ignore TrainAndTest/test_egs.py \
       --model ${model} \
       --resnet-size ${resnet_size} \
       --train-dir ${lstm_dir}/data/${train_set}/egs/${feat_type}/${dev_subset}_fb${input_dim} \
       --train-extract-dir ${lstm_dir}/data/${train_set}/${feat_type}/${dev_subset}_fb${input_dim} \
       --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_fb${input_dim}/trials_dir \
       --train-trials trials_2w \
       --trials ${trials} \
       --valid-dir ${lstm_dir}/data/${train_set}/egs/${feat_type}/${dev_subset}_fb${input_dim}_valid \
       --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${test_subset}_${feat} \
       --feat-format kaldi \
       --nj 4 \
       --input-norm ${input_norm} --input-dim ${input_dim} \
       --kernel-size ${kernel} \
       --downsample ${downsample} \
       --channels 16,32,64,128 \
       --fast none1 --stride 2,1 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 --avg-size 5 \
       --encoder-type ${encoder_type} \
       --alpha 0 \
       --loss-type ${loss} --margin 0.2 --s 30 \
       --test-input fix \
       --remove-vad \
       --resume Data/checkpoint/ThinResNet18/cnceleb/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wd5e4_var/checkpoint_60.pth \
       --xvector-dir Data/xvector/ThinResNet18/cnceleb/klfb_egs_baseline/arcdist_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wd5e4_var/${test_set}_${test_subset}_fix \
       --frame-shift 300 \
       --num-frames 300 \
       --gpu-id 1 \
       --verbose 2 \
       --extract \
       --cos-sim
   done

# official trials
# dev1
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   13.5153   |   0.1367    |   0.6380    |    0.7348    | 20220310 15:07:10 | em256 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   13.5597   |   0.1398    |   0.6392    |    0.7374    | 20220310 15:52:57 | em512 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   11.9840   |   0.3172    |   0.6557    |    0.7721    | 20220310 15:27:22 | em512 arcdist
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   11.6900   |   0.8214    |   0.6140    |    0.7255    | 20220327 01:01:11 | em512 arcdist as-norm
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# dev1+2
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   10.2696   |   0.1820    |   0.5689    |    0.6757    | 20220408 17:05:59 | em512 arcdist
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   10.1864   |   1.6674    |   0.5659    |    0.6789    | 20220408 17:05:59 | em512 arcdist as-norm

# trials eval min 4s
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   10.5546   |   0.1633    |   0.5209    |    0.6148    | 20220310 15:12:26 | em256 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   10.4203   |   0.1684    |   0.5187    |    0.6150    | 20220310 15:58:32 | em512 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |    9.3050   |   0.3394    |   0.5346    |    0.6477    | 20220310 15:32:27 | em512 arcdist
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

# trials eval max 4s
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   16.5088   |   0.1181    |   0.7600    |    0.8670    | 20220310 16:06:42 | em256 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   16.5088   |   0.1181    |   0.7600    |    0.8670    | 20220310 16:01:33 | em512 arcsoft
#+-------------------+-------------+-------------+-------------+--------------+-------------------+
#|   cnceleb-test    |   14.6505   |   0.3004    |   0.7877    |    0.9076    | 20220310 15:42:10 | em512 arcdist
#+-------------------+-------------+-------------+-------------+--------------+-------------------+

       # --verbose 2 \
#              --extract \


 # for s in adve_adve adve_spee dram_reci ente_reci inte_sing live_vlog play_sing sing_vlog adve_dram adve_vlog dram_sing ente_sing inte_spee movi_movi play_spee spee_spee adve_ente dram_spee ente_spee inte_vlog movi_play play_vlog spee_vlog adve_inte dram_dram dram_vlog ente_vlog live_live movi_reci reci_reci vlog_vlog adve_live dram_ente ente_ente inte_inte live_movi movi_sing reci_sing adve_movi dram_inte ente_inte inte_live live_play movi_spee reci_spee adve_play dram_live ente_live inte_movi live_reci movi_vlog reci_vlog adve_reci dram_movi ente_movi inte_play live_sing play_play sing_sing adve_sing dram_play ente_play inte_reci live_spee play_reci sing_spee; do
# for s in adad adpl drdr drre enli envl insi lire more plsi revl vlvl addr adre dren drsi enmo inin insp lisi mosi plsp sisi aden adsi drin drsp enpl inli invl lisp mosp plvl sisp adin adsp drli drvl enre inmo lili livl movl rere sivl adli advl drmo enen ensi inpl limo momo plpl resi spsp admo drpl enin ensp inre lipl mopl plre resp spvl; do
#   python -W ignore TrainAndTest/test_egs.py \
#     --model ${model} --resnet-size 14 \
#     --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#     --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#     --train-trials trials_2w \
#     --trials subtrials/trials_${s} \
#     --score-suffix ${s} \
#     --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#     --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#     --feat-format kaldi \
#     --input-norm ${input_norm} \
#     --input-dim 40 \
#     --channels 512,512,512,512,1500 \
#     --context 5,3,3,5 \
#     --nj 12 \
#     --alpha 0 \
#     --margin 0.2 --s 30 \
#     --stride 1 \
#     --block-type ${block_type} \
#     --embedding-size ${embedding_size} \
#     --loss-type ${loss} \
#     --encoder-type STAP \
#     --test-input fix \
#     --remove-vad \
#     --frame-shift 300 \
#     --xvector-dir Data/xvector/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/${test_set}_dev_epoch50_fix \
#     --resume Data/checkpoint/TDNN_v5/cnceleb/klfb_egs_baseline/arcsoft/Mean_STAP_em512_wd5e4_var/checkpoint_50.pth \
#     --gpu-id 0 \
#     --extract \
#     --cos-sim
# done

#  python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials trials_640w \
#      --score-suffix dev_640w \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --nj 12 \
#      --alpha 0 \
#      --margin 0.15 --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --loss-type ${loss} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
#      --gpu-id 1 \
#      --verbose 2 \
#      --cos-sim

#  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
#  for s in vlog; do
##    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
#    python -W ignore TrainAndTest/test_egs.py \
#      --model ${model} --resnet-size 14 \
#      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
#      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --trials subtrials/trials_vlog_${s} \
#      --score-suffix vl${s} \
#      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
#      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/dev_${feat} \
#      --feat-format kaldi --nj 12 \
#      --input-norm ${input_norm} --input-dim 40 \
#      --channels 512,512,512,512,1500 \
#      --context 5,3,3,5 \
#      --alpha 0 \
#      --loss-type ${loss} --margin 0.15 --s 30 \
#      --stride 1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --encoder-type STAP \
#      --test-input fix \
#      --remove-vad \
#      --frame-shift 300 \
#      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/${test_set}_dev_epoch60_fix \
#      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_revg/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_step5_domain2/checkpoint_60.pth \
#      --gpu-id 1 \
#      --extract \
#      --cos-sim
#  done
  exit
#  for s in vlog; do
#    wc -l data/cnceleb/dev/subtrials/trials_vlog_${s}
#  done
fi

if [ $stage -le 400 ]; then
  feat_type=klfb feat=fb40
  loss=arcsoft
  model=TDNN_v5
  encod=STAP
  dataset=aishell2
  test_set=aishell2 subset=test
  input_dim=40
  input_norm=Mean
  embedding_size=512

  # Training set: aishell2 40-dimensional log fbanks kaldi  Loss: arcsoft
  # Cosine Similarity
  #|     Test Set      |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
  #|     vox1-test     |   21.5960   |   0.2504    |   0.8862    |    0.9153    | 20211213 16:16:06 |
  #|   cnceleb-test    |   25.9654   |   0.2082    |   0.8930    |    0.9493    | 20211213 16:21:13 |
  #|   aishell2-test   |   1.5800    |   0.2139    |   0.3112    |    0.5945    | 20211213 16:22:51 |

  for test_set in vox1 cnceleb aishell2; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${dataset}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi --nj 12 \
      --input-norm ${input_norm} --input-dim ${input_dim} --remove-vad \
      --embedding-size ${embedding_size} \
      --encoder-type ${encod} \
      --channels 512,512,512,512,1500 \
      --stride 1,1,1,1 \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --test-input var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/TDNN_v5/aishell2/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_STAP_em512_wd5e4_var/${test_set}_${subset}_epoch_50_var \
      --resume Data/checkpoint/TDNN_v5/aishell2/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_STAP_em512_wd5e4_var/checkpoint_50.pth \
      --gpu-id 1 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 500 ]; then
  feat_type=klfb feat=log
  model=ThinResNet
  loss=arcsoft
  scheduler=rop optimizer=sgd
  input_dim=40 input_norm=Mean

  encoder_type=ASTP2
  alpha=0
  datasets=aidata testset=aidata
#  test_subset=
  block_type=seblock red_ratio=2
#  encoder_type=None
  embedding_size=256
  batch_size=256
#  resnet_size=18 10
#  sname=dev #dev_aug_com
  sname=train #_aug_com
  downsample=k1
  test_subset=test
  mask_layer=baseline
  dp=0.1
  avg_size=5
  fast=none1
  chn=16
#        --downsample ${downsample} \
#      --trials trials_20w \
  for resnet_size in 34 ;do
        echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
  for select_score in loss_part ; do
  for coreset_percent in 0.25 0.5 0.75 ; do
  for seed in 123456 ; do
#  for lamda_beta in 0.2 0.5 1 2 ; do
  for lamda_beta in 1; do

    if [ $resnet_size -le 34 ];then
      expansion=1
#      batch_size=256
    else
      expansion=2
    fi

    if [ $expansion -eq 1 ]; then
      exp_str=
    else
      exp_str=_exp${expansion}
    fi

    if [ $chn -eq 16 ]; then
        channels=16,32,64,128
        chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    elif [ $chn -eq 64 ]; then
      channels=64,128,256,512
      chn_str=chn64_
    fi

    at_str=
    if [[ $mask_layer == attention* ]];then
      at_str=_${weight}
    elif [ "$mask_layer" = "drop" ];then
      at_str=_${weight}_dp${weight_p}s${scale}
    fi

    if [ $dp = 0.25 ];then
      dp_str=25
    elif [ $dp = 0.2 ];then
      dp_str=01
    elif [ $dp = 0.125 ];then
      dp_str=125
    elif [ $dp = 0.1 ];then
      dp_str=01
    fi
    # ${input_dim}
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_red${red_ratio}${exp_str}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wd5e4_vares_bashuf2_core/percent${coreset_percent}_${select_score}/${seed}
    #_dist_mani-1_lamda2/${seed}
    #_mixup${lamda_beta}_2/${seed}
#    _manifold_0.01
    #${input_dim}

    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_fb40 \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb40/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_fb40 \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/${test_subset}_fb40 --trials trials \
      --feat-format kaldi --nj 8 \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --mask-layer ${mask_layer} --init-weight vox2_rcf \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} --red-ratio ${red_ratio} --expansion ${expansion} \
      --embedding-size ${embedding_size} \
      --kernel-size 5,5 --stride 2,1 --channels ${channels} \
      --downsample ${downsample} --fast ${fast} \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --time-dim 1 --avg-size ${avg_size} \
      --dropout-p ${dp} \
      --xvector-dir Data/xvector/${model_dir}/${testset}_${test_subset}_var \
      --resume Data/checkpoint/${model_dir}/best.pth \
      --gpu-id 0 \
      --remove-vad --test-input var \
      --verbose 0 \
      --cos-sim
  done
  done
  done
  done
  done
  exit

  #|    aidata-test    |  5.29±0.09  |             | 0.4862±0.0133 | 0.8015±0.0142 | baseline
  #|    aidata-test    |  5.10±0.06  |             | 0.4740±0.0082 | 0.8181±0.0083 | manifold_mixup 0.01

fi
