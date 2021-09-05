#!/usr/bin/env bash

stage=22
waited=0
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
while [ $(ps 15414 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  for model in LoResNet10; do
    python Lime/output_extract.py \
      --model ${model} \
      --epochs 19 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect/soft \
      --extract-path Lime/${model} \
      --dropout-p 0.5 \
      --sample-utt 500

  done
fi

if [ $stage -le 1 ]; then
  #  for model in LoResNet10 ; do
  #  python Lime/output_extract.py \
  #    --model LoResNet10 \
  #    --start-epochs 36 \
  #    --epochs 36 \
  #    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
  #    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
  #    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
  #    --loss-type center \
  #    --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect_cmvn/center_dp25 \
  #    --extract-path Data/gradient \
  #    --dropout-p 0 \
  #    --gpu-id 0 \
  #    --embedding-size 1024 \
  #    --sample-utt 2000

  python Lime/output_extract.py \
    --model LoResNet10 \
    --start-epochs 24 \
    --epochs 24 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --loss-type soft \
    --check-path Data/checkpoint/LoResNet10/spect/soft_wcmvn \
    --extract-path Data/gradient/LoResNet10/spect/soft_wcmvn \
    --dropout-p 0.25 \
    --gpu-id 1 \
    --embedding-size 128 \
    --sample-utt 5000

  for loss in amsoft center; do
    python Lime/output_extract.py \
      --model LoResNet10 \
      --start-epochs 38 \
      --epochs 38 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --loss-type ${loss} \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_wcmvn \
      --extract-path Data/gradient/LoResNet10/spect/${loss}_wcmvn \
      --dropout-p 0.25 \
      --s 15 \
      --margin 0.35 \
      --gpu-id 1 \
      --embedding-size 128 \
      --sample-utt 5000
  done
fi

#stage=2
if [ $stage -le 2 ]; then
  model=ExResNet34
  datasets=vox1
  #  feat=fb64_wcmvn
  #  loss=soft
  #  python Lime/output_extract.py \
  #      --model ${model} \
  #      --start-epochs 30 \
  #      --epochs 30 \
  #      --resnet-size 34 \
  #      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64_wcmvn \
  #      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64_wcmvn \
  #      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
  #      --loss-type ${loss} \
  #      --stride 1 \
  #      --remove-vad \
  #      --kernel-size 3,3 \
  #      --check-path Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_var \
  #      --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
  #      --dropout-p 0.0 \
  #      --gpu-id 0 \
  #      --embedding-size 128 \
  #      --sample-utt 10000

  feat=fb64_wcmvn
  loss=soft
  python Lime/output_extract.py \
    --model ExResNet34 \
    --start-epochs 30 \
    --epochs 30 \
    --resnet-size 34 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --loss-type ${loss} \
    --stride 1 \
    --remove-vad \
    --kernel-size 3,3 \
    --check-path Data/checkpoint/ExResNet/spect/soft \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_kaldi \
    --dropout-p 0.0 \
    --gpu-id 1 \
    --time-dim 1 \
    --avg-size 1 \
    --embedding-size 128 \
    --sample-utt 5000
fi

#stage=100
if [ $stage -le 3 ]; then
  model=ResNet20
  datasets=vox1
  feat=spect_256_wcmvn
  loss=soft
  python Lime/output_extract.py \
    --model ${model} \
    --start-epochs 24 \
    --epochs 24 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257_wcmvn \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257_wcmvn \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --loss-type ${loss} \
    --check-path Data/checkpoint/ResNet20/spect_257_wcmvn/soft_dp0.5 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_wcmvn \
    --dropout-p 0.5 \
    --gpu-id 1 \
    --embedding-size 128 \
    --sample-utt 5000
fi

if [ $stage -le 4 ]; then
  model=LoResNet
  train_set=vox2
  test_set=vox1
  feat=log
  loss=arcsoft
  resnet_size=8
  encoder_type=None
  embedding_size=256
  block_type=cbam
  kernel=5,7
  python Lime/output_extract.py \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --start-epochs 40 \
    --epochs 41 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox2/spect/dev_log \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/test_log \
    --input-norm Mean \
    --kernel-size ${kernel} \
    --stride 2,3 \
    --channels 64,128,256 \
    --encoder-type ${encoder_type} \
    --block-type ${block_type} \
    --time-dim 1 \
    --avg-size 4 \
    --embedding-size ${embedding_size} \
    --alpha 0 \
    --loss-type ${loss} \
    --check-path Data/checkpoint/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57 \
    --extract-path Data/gradient/LoResNet8/vox2/spect_egs/arcsoft/None_cbam_dp05_em256_k57 \
    --dropout-p 0.5 \
    --gpu-id 1 \
    --sample-utt 5000

  exit
fi

if [ $stage -le 12 ]; then
  model=ThinResNet
  datasets=vox1
  feat=fb64
  loss=soft
  python Lime/output_extract.py \
    --model ThinResNet \
    --start-epochs 22 \
    --epochs 23 \
    --resnet-size 34 \
    --train-dir ${lstm_dir}/data/${datasets}/pyfb/dev_${feat} \
    --test-dir ${lstm_dir}/data/${datasets}/pyfb/test_${feat} \
    --loss-type ${loss} \
    --stride 1 \
    --remove-vad \
    --kernel-size 5,5 \
    --encoder-type None \
    --check-path Data/checkpoint/ThinResNet34/vox1/fb64_None/soft \
    --extract-path Data/gradient/ThinResNet34/vox1/fb64_None/soft \
    --dropout-p 0.0 \
    --gpu-id 0 \
    --time-dim 1 \
    --avg-size 1 \
    --embedding-size 128 \
    --sample-utt 5000
fi
#stage=300
#stage=1000

if [ $stage -le 20 ]; then
  model=LoResNet10
  datasets=timit
  feat=spect
  loss=soft

  #  python Lime/output_extract.py \
  #    --model LoResNet10 \
  #    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
  #    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
  #    --start-epochs 15 \
  #    --check-path Data/checkpoint/LoResNet10/timit_spect/soft_fix \
  #    --epochs 15 \
  #    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
  #    --sample-utt 1500 \
  #    --embedding-size 128 \
  #    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_fix \
  #    --model ${model} \
  #    --channels 4,16,64 \
  #    --dropout-p 0.25

  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/timit_spect/soft_var \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 10000 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
    --model ${model} \
    --channels 4,16,64 \
    --dropout-p 0.25
fi

if [ $stage -le 21 ]; then
  model=LoResNet
  dataset=timit
  train_set=timit
  test_set=timit
  feat_type=spect
  feat=log
  loss=soft
  resnet_size=8
  encoder_type=None
  embedding_size=128
  block_type=basic
  kernel=5,5
  python Lime/output_extract.py \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --start-epochs 12 \
    --epochs 12 \
    --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/train_${feat} \
    --train-set-name timit \
    --test-set-name timit \
    --test-dir ${lstm_dir}/data/${dataset}/${feat_type}/test_${feat} \
    --input-norm None \
    --kernel-size ${kernel} \
    --stride 2 \
    --channels 4,16,64 \
    --encoder-type ${encoder_type} \
    --block-type ${block_type} \
    --time-dim 1 \
    --avg-size 4 \
    --embedding-size ${embedding_size} \
    --alpha 10.8 \
    --loss-type ${loss} \
    --dropout-p 0.5 \
    --check-path Data/checkpoint/LoResNet8/timit/spect_egs_log/soft_dp05 \
    --extract-path Data/gradient/LoResNet8/timit/spect_egs_log/soft_dp05/epoch_12_var_50 \
    --gpu-id 1 \
    --sample-utt 50
  exit
fi

if [ $stage -le 22 ]; then
  model=LoResNet
  dataset=vox2
  train_set=vox2
  test_set=vox1
  feat_type=klsp
  feat=log
  loss=arcsoft
  resnet_size=8
  encoder_type=None
  embedding_size=256
  block_type=cbam
  kernel=5,5
  echo -e "\n\033[1;4;31m stage${stage} Training ${model}_${encoder_type} in ${train_set}_${test_set} with ${loss}\033[0m\n"

  python Lime/output_extract.py \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --start-epochs 50 \
    --epochs 50 \
    --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev \
    --train-set-name vox1 \
    --test-set-name vox1 \
    --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/test \
    --input-norm Mean \
    --kernel-size ${kernel} \
    --stride 2 \
    --channels 64,128,256 \
    --encoder-type ${encoder_type} \
    --block-type ${block_type} \
    --time-dim 1 \
    --avg-size 4 \
    --embedding-size ${embedding_size} \
    --alpha 0 \
    --loss-type ${loss} \
    --dropout-p 0.1 \
    --check-path Data/checkpoint/LoResNet8/vox2/klsp_egs_baseline/arcsoft/Mean_cbam_None_dp01_alpha0_em256_var \
    --extract-path Data/gradient/LoResNet8/vox2/klsp_egs_baseline/arcsoft/Mean_cbam_None_dp01_alpha0_em256_var/epoch_50_var_50 \
    --gpu-id 1 \
    --sample-utt 120
  exit
fi

#stage=500

if [ $stage -le 30 ]; then
  model=LoResNet10
  datasets=libri
  feat=spect_noc
  loss=soft

  #  python Lime/output_extract.py \
  #    --model LoResNet10 \
  #    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
  #    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
  #    --start-epochs 15 \
  #    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss} \
  #    --epochs 15 \
  #    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
  #    --sample-utt 4000 \
  #    --embedding-size 128 \
  #    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
  #    --model ${model} \
  #    --channels 4,32,128 \
  #    --dropout-p 0.25

  #  python Lime/output_extract.py \
  #    --model LoResNet10 \
  #    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
  #    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
  #    --start-epochs 15 \
  #    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss}_var \
  #    --epochs 15 \
  #    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
  #    --sample-utt 4000 \
  #    --embedding-size 128 \
  #    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
  #    --model ${model} \
  #    --channels 4,32,128 \
  #    --dropout-p 0.25
  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss} \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 4000 \
    --alpha 9.8 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
    --model ${model} \
    --channels 4,16,64 \
    --dropout-p 0.25
fi

if [ $stage -le 40 ]; then
  model=TDNN
  feat=fb40
  datasets=vox1
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python Lime/output_extract.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${datasets}/pyfb/dev_${feat} \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${datasets}/pyfb/test_${feat} \
      --nj 14 \
      --start-epochs 20 \
      --epochs 21 \
      --model ${model} \
      --embedding-size 128 \
      --sample-utt 5000 \
      --feat-dim 40 \
      --remove-vad \
      --check-path Data/checkpoint/${model}/${datasets}/${feat}_STAP/soft \
      --extract-path Data/gradient/${model}/${datasets}/${feat}_STAP/soft \
      --loss-type soft \
      --gpu-id 0
  done
fi

stage=1000
if [ $stage -le 50 ]; then
  model=SiResNet34
  feat=fb40_wcmvn
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python Lime/output_extract.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64 \
      --nj 14 \
      --start-epochs 21 \
      --epochs 21 \
      --model ${model} \
      --embedding-size 128 \
      --sample-utt 5000 \
      --feat-dim 64 \
      --kernel-size 3,3 \
      --stride 1 \
      --input-length fix \
      --remove-vad \
      --mvnorm \
      --check-path Data/checkpoint/SiResNet34/vox1/fb64_cmvn/soft \
      --extract-path Data/gradient/SiResNet34/vox1/fb64_cmvn/soft \
      --loss-type soft \
      --gpu-id 1
  done
fi

if [ $stage -le 60 ]; then
  model=LoResNet10
  feat=spect
  dataset=cnceleb
  for loss in soft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python Lime/output_extract.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/eval \
      --nj 14 \
      --start-epochs 24 \
      --epochs 24 \
      --model ${model} \
      --embedding-size 128 \
      --sample-utt 2500 \
      --feat-dim 161 \
      --kernel-size 3,3 \
      --channels 64,128,256,256 \
      --resnet-size 18 \
      --check-path Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --extract-path Data/gradient/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --loss-type soft \
      --gpu-id 1
  done
fi

if [ $stage -le 61 ]; then
  model=LoResNet
  feat=spect
  dataset=timit
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  for loss in arcsoft; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python Lime/output_extract.py \
      --train-dir ${lstm_dir}/data/${dataset}/spect/train_log \
      --test-dir ${lstm_dir}/data/${dataset}/spect/test_log \
      --nj 12 \
      --start-epochs 12 \
      --epochs 12 \
      --model ${model} \
      --embedding-size 128 \
      --sample-utt 2500 \
      --feat-dim 161 \
      --kernel-size 5,5 \
      --channels 4,16,64 \
      --resnet-size 8 \
      --check-path Data/checkpoint/${model}8/${dataset}/spect_egs_log/${loss}_dp05 \
      --extract-path Data/gradient/${model}8/${dataset}/spect_egs_log/${loss}_dp05 \
      --loss-type ${loss} \
      --gpu-id 0
  done
fi

#stage=100
if [ $stage -le 62 ]; then
  dataset=timit
  for numframes in 1500; do
    echo -e "\033[31m==> num of frames per speaker : ${numframes} \033[0m"
    python Lime/fratio_extract.py \
      --extract-frames \
      --file-dir ${lstm_dir}/data/${dataset}/spect/train_power \
      --out-dir Data/fratio/${dataset}/dev_power \
      --nj 14 \
      --input-per-spks ${numframes} \
      --extract-frames \
      --feat-dim 161
  done
fi

if [ $stage -le 80 ]; then
  dataset=vox1
  for numframes in 15000; do
    echo -e "\033[31m==> num of frames per speaker : ${numframes} \033[0m"
    python Lime/fratio_extract.py \
      --extract-frames \
      --file-dir ${lstm_dir}/data/${dataset}/spect/dev_power \
      --out-dir Data/fratio/vox1/dev_power \
      --nj 14 \
      --input-per-spks ${numframes} \
      --extract-frames \
      --feat-dim 161
  done
fi
