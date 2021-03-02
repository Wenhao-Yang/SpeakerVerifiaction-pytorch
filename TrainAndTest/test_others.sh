#!/usr/bin/env bash

stage=101
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/all_wcmvn \
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
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
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
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
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
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
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

#stage=200
if [ $stage -le 15 ]; then
  model=TDNN
  #  feat=fb40
  #  for loss in soft ; do
  #    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
  #    python TrainAndTest/test_egs.py \
  #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_no_sil \
  #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_no_sil \
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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
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

#stage=100
if [ $stage -le 30 ]; then
  model=ResNet20
  feat=spect_wcmvn
  datasets=vox
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} fix length \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257_wcmvn \
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
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64_wcmvn \
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
    #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
    #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
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

if [ $stage -le 50 ]; then
  #  for loss in soft asoft ; do
  model=SiResNet34
  datasets=vox1
  feat=fb64_mvnorm
  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64 \
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

if [ $stage -le 55 ]; then
  #  for loss in soft asoft ; do
  model=GradResNet
  datasets=vox1
  feat=fb64_mvnorm
  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/dev_power \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/test_power \
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

if [ $stage -le 80 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  encod=None
  dataset=vox1
  for loss in arcsoft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Testing with ${loss} \033[0m\n"
    python -W ignore TrainAndTest/test_egs.py \
      --model TDNN_v5 \
      --train-dir ${lstm_dir}/data/vox2/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --input-length var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox2/spect_STAP_v2/arcsoft_100ce/emsize512_inputMean/checkpoint_53.pth/epoch_53_var \
      --resume Data/checkpoint/TDNN_v5/vox2/spect_STAP_v2/arcsoft_100ce/emsize512_inputMean/checkpoint_53.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi

if [ $stage -le 81 ]; then
  feat_type=spect
  feat=log
  loss=arcsoft
  encod=None
  dataset=vox1
  block_type=Basic
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
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
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
      --frame-shift 300 \
      --dropout-p 0.5 \
      --xvector-dir Data/xvector/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57/epoch_17_var \
      --resume Data/checkpoint/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57/checkpoint_17.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi

if [ $stage -le 100 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=18
  #  loss=soft
  encod=None
  transform=None
  loss_ratio=0.01
  alpha=0
  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model}_${resnet_size} in army with ${loss} kernel 5,5 \033[0m\n"

    python TrainAndTest/test_egs.py \
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_dev_8k_v4 \
      --train-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_dev_8k_v4 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2/trials_dir \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v4 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_valid_8k_v4 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 256 \
      --nj 10 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --lr 0.1 \
      --input-dim 81 \
      --fast \
      --mask-layer freq \
      --mask-len 20 \
      --stride 1 \
      --milestones 8,14,20 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_x4/spect_egs/${loss}/${encod}_dp25_eb256_${alpha}_fast_${transform}_mask20 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_x4/spect_egs/${loss}/${encod}_dp25_eb256_${alpha}_fast_${transform}_mask20/checkpoint_29.pth \
      --channels 32,64,128,256 \
      --embedding-size 256 \
      --transform ${transform} \
      --encoder-type ${encod} \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha ${alpha} \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --loss-ratio ${loss_ratio} \
      --set-ratio 1.0 \
      --weight-decay 0.0005 \
      --dropout-p 0.25 \
      --gpu-id 0,1 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi

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
      --input-norm None \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 128 \
      --loss-type ${loss} \
      --encoder-type None \
      --block-type ${block_type} \
      --kernel-size 5,5 \
      --stride 2 \
      --channels 64,128,256,256 \
      --alpha 0 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --input-length var \
      --frame-shift 300 \
      --dropout-p 0.25 \
      --xvector-dir Data/xvector/LoResNet10/army_v1/spect_egs/soft_dp25/epoch_20_var \
      --resume Data/checkpoint/LoResNet10/army_v1/spect_egs/soft_dp25/checkpoint_20.pth \
      --gpu-id 0 \
      --cos-sim
  done
fi
