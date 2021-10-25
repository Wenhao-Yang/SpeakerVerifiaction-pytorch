#!/usr/bin/env bash

stage=83
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
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_egs.py \
#      --train-dir ${lstm_dir}/data/Vox1_spect/dev \
#      --test-dir ${lstm_dir}/data/Vox1_spect/test \
#      --nj 12 \
#      --model ${model} \
#      --resume Data/checkpoint/LoResNet10/spect_cmvn/${loss}_dp25/checkpoint_36.pth \
#      --loss-type ${loss} \
#      --num-valid 2 \
#      --gpu-id 1
#  done
fi

if [ $stage -le 6 ]; then
  model=LoResNet10
  #  --resume Data/checkpoint/LoResNet10/spect/${loss}_dp25_128/checkpoint_24.pth \
  #  for loss in soft ; do
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    #    python TrainAndTest/test_egs.py \
    #      --train-dir ${lstm_dir}/data/Vox1_spect/dev_wcmvn \
    #      --test-dir ${lstm_dir}/data/Vox1_spect/test_wcmvn \
    #      --nj 12 \
    #      --model ${model} \
    #      --channels 64,128,256,512 \
    #      --resnet-size 10 \
    #      --extract \
    #      --kernel-size 3,3 \
    #      --embedding-size 128 \
    #      --resume Data/checkpoint/LoResNet10/spect/soft_dp05/checkpoint_36.pth \
    #      --xvector-dir Data/xvector/LoResNet10/spect/soft_dp05 \
    #      --loss-type ${loss} \
    #      --trials trials.backup \
    #      --num-valid 0 \
    #      --gpu-id 0
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

#  model=LoResNet10
#  for loss in soft ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_egs.py \
#      --train-dir ${lstm_dir}/data/Vox1_spect/dev_wcmvn \
#      --test-dir ${lstm_dir}/data/Vox1_spect/test_wcmvn \
#      --nj 12 \
#      --model ${model} \
#      --channels 64,128,256 \
#      --resnet-size 8 \
#      --kernel-size 5,5 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/LoResNet8/spect/soft_wcmvn/checkpoint_24.pth \
#      --extract \
#      --xvector-dir Data/xvector/LoResNet8/spect/soft_wcmvn \
#      --loss-type ${loss} \
#      --trials trials.backup \
#      --num-valid 0 \
#      --gpu-id 0
#  done

#  for loss in center ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_egs.py \
#      --train-dir ${lstm_dir}/data/Vox1_spect/dev \
#      --test-dir ${lstm_dir}/data/Vox1_spect/test \
#      --nj 12 \
#      --model ${model} \
#      --resume Data/checkpoint/LoResNet10/spect_cmvn/${loss}_dp25/checkpoint_36.pth \
#      --loss-type ${loss} \
#      --num-valid 2 \
#      --gpu-id 1
#  done
fi

#python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#    --check-path Data/checkpoint/LoResNet10/spect/amsoft \
#    --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
#    --loss-type amsoft \
#    --lr 0.01 \
#    --epochs 10

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
      --input-length var \
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
      --input-length fix \
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
      --nj 12 \
      --epochs 18 \
      --model ${model} \
      --resnet-size 8 \
      --inst-norm \
      --embedding-size 128 \
      --feat-dim 161 \
      --valid \
      --input-length fix \
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
      --input-length fix \
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
      --input-length fix \
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

if [ $stage -le 76 ]; then
  feat_type=pyfb
  feat=fb40
  loss=soft
  model=TDNN_v5
  encod=None
  dataset=vox1
  subset=test
  test_set=sitw

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
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 40 \
      --nj 12 \
      --embedding-size 256 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --input-length var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4_var/${test_set}_${subset}_epoch_40_var \
      --resume Data/checkpoint/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wd5e4_var/checkpoint_40.pth \
      --gpu-id 1 \
      --remove-vad \
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
      --input-length var \
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
#      --input-length var \
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
#      --input-length var \
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
      --input-length var \
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
      --input-length var \
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
      --input-length var \
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
      --input-length var \
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
      --input-length var \
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

  #  for loss in soft arcsoft; do # 32,128,512; 8,32,128
  for s in advertisement drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
    echo -e "\n\033[1;4;31m Stage${stage}: Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 14 \
      --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --trials trials_${s} \
      --score-suffix ${s} \
      --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/test_${feat} \
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
      --input-length fix \
      --remove-vad \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/${test_set}_test_epoch60_fix \
      --resume Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_baseline/${loss}/featfb40_ws25_inputMean_STAP_em256_wde3_var/checkpoint_60.pth \
      --gpu-id 0 \
      --extract \
      --cos-sim
  done
  exit
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
      --input-length fix \
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
      --input-length var \
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
      --input-length var \
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
      --input-length var \
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
  encod=None
  alpha=0
  datasets=vox2
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8

  for sname in dev ; do
    echo -e "\n\033[1;4;31mStage ${stage}: Testing ${model}_${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model ${model} \
      --resnet-size 8 \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
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
      --input-length var \
      --dropout-p 0.1 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/Mean_cbam_None_dp01_alpha0_em256_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/Mean_cbam_None_dp01_alpha0_em256_var/checkpoint_61.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#|   Test Set   |   EER (%)   |  Threshold  | MinDCF-0.01 | MinDCF-0.001 |       Date        |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#|  vox2-test   |   3.2715%   |   0.2473    |   0.3078    |    0.4189    | 20210818 19:07:02 |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
#|  vox2-test   |   2.8367%   |   0.2615    |   0.2735    |    0.4051    | 20210818 19:11:29 |
#+--------------+-------------+-------------+-------------+--------------+-------------------+
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
      --input-length var \
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
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=None
  alpha=0
  datasets=vox1
  testset=vox1
#  test_subset=
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  test_subset=test
  input_norm=Mean

  for weight in mel clean aug vox2 ; do
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
      --mask-layer attention \
      --score-suffix ${weight} \
      --init-weight ${weight} \
      --encoder-type ${encod} \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2,2 \
      --channels 16,32,64 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --input-length var \
      --dropout-p 0.125 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model}${resnet_size}/${datasets}/${feat_type}_egs_attention/${loss}/${input_norm}_${block_type}_${encod}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn16_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_attention/${loss}/${input_norm}_${block_type}_${encod}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn16_wd5e4_var/checkpoint_50.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi
if [ $stage -le 96 ]; then
  feat_type=klsp
  model=LoResNet
  feat=log
  loss=arcsoft
  encod=None
  alpha=0
  datasets=vox1
  testset=vox1
#  test_subset=
  block_type=cbam
  encoder_type=None
  embedding_size=256
  resnet_size=8
#  sname=dev #dev_aug_com
  sname=dev #_aug_com
  test_subset=test
  input_norm=Mean

  for weight in None ; do
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
      --channels 16,32,64 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --input-length var \
      --dropout-p 0.125 \
      --time-dim 1 \
      --avg-size 4 \
      --xvector-dir Data/xvector/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encod}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encod}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var/checkpoint_50.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi


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
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/spect/aishell2_dev_8k_v4 \
      --train-dir-b ${lstm_dir}/data/${datasets}/spect/vox_dev_8k_v4 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2/trials_dir \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v4 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_valid_8k_v4 \
      --test-dir ${lstm_dir}/data/magic/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 128 \
      --nj 10 \
      --lr 0.1 \
      --input-dim 81 \
      --fast \
      --mask-layer freq \
      --mask-len 20 \
      --stride 1 \
      --xvector-dir Data/xvector/MultiResNet18/army_x4/spect_egs_None/soft/dp25_b256_13_fast_None_mask/epoch_36_var_magic \
      --resume Data/checkpoint/MultiResNet18/army_x4/spect_egs_None/soft/dp25_b256_13_fast_None_mask/checkpoint_36.pth \
      --channels 32,64,128,256 \
      --embedding-size 128 \
      --transform ${transform} \
      --encoder-type ${encod} \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha ${alpha} \
      --margin 0.3 \
      --input-length var \
      --s 30 \
      --m 3 \
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
  feat_type=spect
  feat=log
  loss=arcsoft
  encod=None
  dataset=army_v1
  block_type=None
  for loss in soft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model LoResNet \
      --resnet-size 10 \
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
      --kernel-size 5,5 \
      --stride 2 \
      --channels 64,128,256,256 \
      --alpha 12 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --input-length var \
      --frame-shift 300 \
      --dropout-p 0.1 \
      --xvector-dir Data/xvector/LoResNet10/army_v1/spect_egs_mean/soft_dp01/epoch_20_var \
      --resume Data/checkpoint/LoResNet10/army_v1/spect_egs_mean/soft_dp01/checkpoint_20.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi
