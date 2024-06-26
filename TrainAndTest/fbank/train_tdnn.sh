#!/usr/bin/env bash

stage=300  # skip to stage x
waited=0
while [ $(ps 18118 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

lstm_dir=/home/yangwenhao/project/lstm_speaker_verification
if [ $stage -le 0 ]; then
  model=ETDNN
  for loss in soft; do
    python TrainAndTest/Fbank/TDNNs/train_etdnn_kaldi.py \
      --train-dir ${lstm_dir}/data/Vox1_pyfb80/dev_kaldi \
      --test-dir ${lstm_dir}/data/Vox1_pyfb80/test_kaldi \
      --check-path Data/checkpoint/${model}/fbank80/soft \
      --resume Data/checkpoint/${model}/fbank80/soft/checkpoint_1.pth \
      --epochs 20 \
      --milestones 10,15 \
      --feat-dim 80 \
      --embedding-size 256 \
      --num-valid 2 \
      --loss-type soft \
      --lr 0.01

  done
fi

#stage=1
if [ $stage -le 5 ]; then
  model=TDNN
  feat=fb40
  for loss in soft; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_var.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40 \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40 \
      --check-path Data/checkpoint/${model}/${feat}/${loss}_norm \
      --resume Data/checkpoint/${model}/${feat}/${loss}_norm/checkpoint_1.pth \
      --batch-size 64 \
      --epochs 16 \
      --milestones 8,12 \
      --feat-dim 40 --remove-vad \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 192 \
      --gpu-id 0 \
      --veri-pairs 9600 \
      --lr 0.01
  done
fi

#stage=100
if [ $stage -le 10 ]; then
  model=ASTDNN
  feat=fb40_wcmvn
  for loss in soft; do
    #    python TrainAndTest/Fbank/TDNNs/train_astdnn_kaldi.py \
    #      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_wcmvn \
    #      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_wcmvn \
    #      --check-path Data/checkpoint/${model}/${feat}/${loss} \
    #      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_1.pth \
    #      --epochs 18 \
    #      --batch-size 128 \
    #      --milestones 9,14  \
    #      --feat-dim 40 \
    #      --embedding-size 128 \
    #      --num-valid 2 \
    #      --loss-type ${loss} \
    #      --input-per-spks 240 \
    #      --lr 0.01

    python TrainAndTest/Fbank/TDNNs/train_tdnn_var.py \
      --model ASTDNN \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_wcmvn \
      --check-path Data/checkpoint/${model}/${feat}/${loss}_svar \
      --resume Data/checkpoint/${model}/${feat}/${loss}_svar/checkpoint_1.pth \
      --epochs 18 \
      --batch-size 128 \
      --milestones 9,14 \
      --feat-dim 40 --remove-vad \
      --embedding-size 512 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 240 \
      --lr 0.01 \
      --gpu-id 1

    python TrainAndTest/test_vox1.py \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_wcmvn \
      --nj 12 \
      --model ASTDNN \
      --embedding-size 512 \
      --feat-dim 40 --remove-vad \
      --resume Data/checkpoint/${model}/${feat}/${loss}_svar/checkpoint_18.pth \
      --loss-type soft \
      --num-valid 2 \
      --gpu-id 1

    python Lime/output_extract.py \
      --model ASTDNN \
      --start-epochs 18 \
      --epochs 18 \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb40_wcmvn \
      --sitw-dir ${lstm_dir}/data/sitw \
      --loss-type soft \
      --remove-vad \
      --check-path Data/checkpoint/${model}/${feat}/${loss}_svar \
      --extract-path Data/gradient/${model}/${feat}/${loss}_svar \
      --gpu-id 1 \
      --embedding-size 512 \
      --sample-utt 5000
  done
fi

#stage=1
if [ $stage -le 15 ]; then
  model=ETDNN
  feat=fb80
  for loss in soft; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_kaldi.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb80 \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb80 \
      --check-path Data/checkpoint/${model}/${feat}/${loss} \
      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_1.pth \
      --batch-size 128 \
      --epochs 20 \
      --milestones 10,14 \
      --feat-dim 80 --remove-vad \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 224 \
      --gpu-id 1 \
      --veri-pairs 9600 \
      --lr 0.01
  done
fi

#stage=100
if [ $stage -le 16 ]; then
  model=ETDNN
  feat=fb80
  for loss in amsoft center; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_kaldi.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/Vox1_pyfb/dev_fb80 \
      --test-dir ${lstm_dir}/data/Vox1_pyfb/test_fb80 \
      --check-path Data/checkpoint/${model}/${feat}/${loss} \
      --resume Data/checkpoint/ETDNN/fb80/soft/checkpoint_20.pth \
      --batch-size 128 \
      --epochs 30 \
      --finetune \
      --milestones 6 \
      --feat-dim 80 --remove-vad \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} --m 4 --margin 0.35 --s 15 \
      --loss-ratio 0.01 \
      --input-per-spks 224 \
      --gpu-id 1 \
      --veri-pairs 9600 \
      --lr 0.001
  done
fi

if [ $stage -le 40 ]; then
  model=TDNN
  resnet_size=34
  datasets=vox1
  feat=fb40
  loss=soft

  for encod in SASP; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/vox1/egs/pyfb/dev_${feat} \
      --valid-dir ${lstm_dir}/data/vox1/egs/pyfb/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/pyfb/test_${feat} \
      --nj 10 --epochs 20 \
      --milestones 10,15 \
      --model ${model} --resnet-size ${resnet_size} \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --input-dim 40 \
      --lr 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat}_${encod}/${loss} \
      --resume Data/checkpoint/${model}/${datasets}/${feat}_${encod}/${loss}/checkpoint_22.pth \
      --input-per-spks 384 \
      --cos-sim \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft \
      --remove-vad

  done
fi


# VoxCeleb1

if [ $stage -le 50 ]; then
  model=TDNN_v4
  datasets=vox1
  feat=fb24_kaldi
  loss=soft

  for encod in STAP; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/vox1/egs/pyfb/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/pyfb/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/pyfb/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/pyfb/test_${feat} \
      --nj 8 \
      --epochs 24 \
      --milestones 8,14,20 \
      --model ${model} \
      --scheduler rop \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --input-dim 24 \
      --lr 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat}_${encod}/${loss} \
      --resume Data/checkpoint/${model}/${datasets}/${feat}_${encod}/${loss}/checkpoint_22.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft \
      --log-interval 10

  done
fi

if [ $stage -le 60 ]; then
  model=TDNN_v5
  datasets=vox1
  feat=log
  feat_type=spect
  loss=soft
  encod=STAP
  embedding_size=256

  for model in TDNN_v4; do
    echo -e "\n\033[1;4;31m Stage ${stage}:Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/vox1/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --nj 8 \
      --epochs 60 \
      --milestones 8,14,20 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size 192 \
      --accu-steps 1 \
      --input-dim 161 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}/${loss}_emsize${embedding_size} \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}/${loss}_emsize${embedding_size}/checkpoint_40.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.3 \
      --s 15 \
      --log-interval 10
  done
fi

if [ $stage -le 70 ]; then
  model=SlimmableTDNN
  datasets=vox1 testsets=vox1
  #  feat=fb24
  feat_type=klfb
  loss=arcsoft
  encoder_type=STAP
  embedding_size=512
  input_dim=40
  input_norm=Mean
  optimizer=sgd scheduler=rop
  mask_layer=baseline

  alpha=0
  weight=vox2_cf
  weight_p=0
  scale=0.2
  batch_size=256
  stat_type=margin
  loss_ratio=1

  for model in SlimmableTDNN  ; do
  for seed in 123456 123457 123458 ;do
  for embedding_size in 512; do
    #    feat=combined
    if [[ $loss == *dist ]];then
      loss_str=_stat${stat_type}${loss_ratio}
    else
      loss_str=
    fi

    model_dir=${model}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encoder_type}_em${embedding_size}_dp00_alpha${alpha}_${loss_str}wd5e4_var3_lr04

    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test_fb${input_dim} \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test_fb${input_dim} \
      --nj 6 \
      --seed ${seed} \
      --shuffle \
      --epochs 50 \
      --batch-size ${batch_size} \
      --patience 3 \
      --milestones 10,20,30,40 \
      --model ${model} \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --early-stopping --early-patience 15 --early-delta 0.0001 --early-meta EER \
      --lr 0.04 \
      --base-lr 0.00000001 \
      --weight-decay 0.0005 \
      --alpha 0 \
      --feat-format kaldi \
      --width-mult-list 0.25,0.5,0.75,1 \
      --embedding-size ${embedding_size} \
      --accu-steps 1 \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1536 \
      --encoder-type ${encoder_type} \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 2,3 \
      --num-valid 2 \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --remove-vad \
      --log-interval 10 \
      --stat-type ${stat_type} \
      --loss-ratio ${loss_ratio}
#  done


#  loss=arcsoft
#  for embedding_size in 512; do
#    mask_layer=drop
#    python -W ignore TrainAndTest/train_egs.py \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
#      --train-trials trials_2w \
#      --shuffle \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_fb${input_dim} \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
#      --nj 16 \
#      --epochs 50 \
#      --patience 3 \
#      --milestones 10,20,30,40 \
#      --model ${model} \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.00000001 \
#      --weight-decay 0.0005 \
#      --alpha 0 \
#      --feat-format kaldi \
#      --embedding-size ${embedding_size} \
#      --batch-size ${batch_size} \
#      --accu-steps 1 \
#      --random-chunk 200 400 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --weight-p ${weight_p} \
#      --scale ${scale} \
#      --input-dim ${input_dim} \
#      --channels 512,512,512,512,1500 \
#      --encoder-type ${encod} \
#      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/input${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_${weight}scale${scale}p${weight_p}_wd5e4_var \
#      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/input${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_${weight}scale${scale}p${weight_p}_wd5e4_var/checkpoint_50.pth \
#      --cos-sim \
#      --dropout-p 0.0 \
#      --veri-pairs 9600 \
#      --gpu-id 0,1 \
#      --num-valid 2 \
#      --loss-type ${loss} \
#      --margin 0.2 \
#      --s 30 \
#      --remove-vad \
#      --log-interval 10
  done
  done
  done
  exit
fi

if [ $stage -le 74 ]; then
  model=TDNN_v5
  datasets=vox1
  #  feat=fb24
  feat_type=klsp
  loss=soft
  encod=STAP
  embedding_size=256
  input_dim=161
  input_norm=Mean
  # _lrr${lr_ratio}_lsr${loss_ratio}
  feat=klsp
#
#  for loss in arcsoft; do
#
#    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
#    python -W ignore TrainAndTest/train_egs.py \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --nj 12 \
#      --epochs 40 \
#      --patience 2 \
#      --milestones 10,20,30 \
#      --model ${model} \
#      --scheduler rop \
#      --weight-decay 0.0005 \
#      --lr 0.1 \
#      --alpha 0 \
#      --feat-format kaldi \
#      --embedding-size ${embedding_size} \
#      --var-input \
#      --batch-size 128 \
#      --accu-steps 1 \
#      --shuffle \
#      --random-chunk 200 400 \
#      --input-dim ${input_dim} \
#      --channels 512,512,512,512,1500 \
#      --encoder-type ${encod} \
#      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var \
#      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_13.pth \
#      --cos-sim \
#      --dropout-p 0.0 \
#      --veri-pairs 9600 \
#      --gpu-id 0,1 \
#      --num-valid 2 \
#      --loss-type ${loss} \
#      --margin 0.2 \
#      --s 30 \
#      --log-interval 10
#  done

#  for loss in arcsoft; do
#
#    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
#    python -W ignore TrainAndTest/train_egs.py \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --nj 12 \
#      --epochs 40 \
#      --patience 2 \
#      --milestones 10,20,30 \
#      --model ${model} \
#      --scheduler rop \
#      --weight-decay 0.0005 \
#      --lr 0.1 \
#      --alpha 0 \
#      --feat-format kaldi \
#      --embedding-size ${embedding_size} \
#      --var-input \
#      --batch-size 128 \
#      --accu-steps 1 \
#      --shuffle \
#      --random-chunk 200 400 \
#      --input-dim ${input_dim} \
#      --channels 256,256,256,256,768 \
#      --encoder-type ${encod} \
#      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_chn256_wd5e4_var \
#      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_chn256_wd5e4_var/checkpoint_13.pth \
#      --cos-sim \
#      --dropout-p 0.0 \
#      --veri-pairs 9600 \
#      --gpu-id 0,1 \
#      --num-valid 2 \
#      --loss-type ${loss} \
#      --margin 0.2 \
#      --s 30 \
#      --log-interval 10
#  done
  loss=arcsoft
  mask_layer=attention
  for weight in mel clean aug vox2; do
#    weight=clean
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --nj 12 \
      --epochs 40 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --channels 256,256,256,256,768 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${encod}_em${embedding_size}_${weight}42_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${encod}_em${embedding_size}_${weight}42_wd5e4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --log-interval 10
  done
  exit
fi

# VoxCeleb2


if [ $stage -le 100 ]; then
  model=TDNN_v5
  datasets=vox2
  #  feat=fb24
  feat_type=klsp
  loss=soft
  encod=STAP
  embedding_size=512
  input_dim=161
  input_norm=Mean
  # _lrr${lr_ratio}_lsr${loss_ratio}

  for loss in arcsoft; do
    feat=fb${input_dim}_ws25
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --nj 12 \
      --epochs 40 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wde4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wde4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --log-interval 10
  done
  exit
fi

if [ $stage -le 101 ]; then
  model=TDNN_v5
  datasets=vox2
  #  feat=fb24
  feat_type=klfb
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_dim=40
  input_norm=Mean
  optimizer=sgd
  scheduler=exp
  # _lrr${lr_ratio}_lsr${loss_ratio}

  for loss in arcsoft; do
    feat=fb${input_dim}
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --nj 16 \
      --epochs 50 \
      --patience 3 \
      --milestones 10,20,30 \
      --model ${model} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --weight-decay 0.0001 \
      --lr 0.1 \
      --base-lr 0.000005 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/input${input_norm}_${encod}_em${embedding_size}_wde4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/input${input_norm}_${encod}_em${embedding_size}_wde4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --remove-vad \
      --log-interval 10
  done
  exit
fi

if [ $stage -le 105 ]; then
  model=ECAPA
  datasets=vox2
  #  feat=fb24
  feat_type=pyfb
  loss=soft
  encod=SASP
  embedding_size=192
  input_dim=40
  input_norm=None

  for loss in arcsoft; do
    feat=fb${input_dim}_ws25
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --shuffle \
      --nj 16 \
      --epochs 60 \
      --patience 3 \
      --milestones 10,20,30,40 \
      --model ${model} \
      --optimizer adam \
      --scheduler cyclic \
      --weight-decay 0.00001 \
      --lr 0.001 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --random-chunk 300 301 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1536 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/feat${feat}_input${input_norm}_${encod}128_em${embedding_size}_wde5_adam \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/feat${feat}_input${input_norm}_${encod}128_em${embedding_size}_wde5_adam/checkpoint_5.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --remove-vad \
      --log-interval 10
  done
  exit
fi

if [ $stage -le 106 ]; then
  model=RET
  datasets=vox2
  feat_type=pyfb
  loss=soft
  encod=STAP
  embedding_size=512
  input_dim=40
  input_norm=Mean
  resnet_size=14

  for loss in arcsoft; do
    feat=fb${input_dim}_ws25
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --nj 16 \
      --epochs 50 \
      --patience 3 \
      --milestones 10,20,30 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --scheduler rop \
      --weight-decay 0.0001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,512,1536 \
      --context 5,5,5 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde4_var/checkpoint_9.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.3 \
      --s 15 \
      --remove-vad \
      --log-interval 10
  done
  exit
fi
if [ $stage -le 107 ]; then
  model=TDNN_v5
  datasets=vox2
  feat=fb40
  feat_type=pyfb
  loss=arcsoft
  encod=STAP
  embedding_size=512

  for model in TDNN_v5; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --fix-length \
      --nj 16 \
      --epochs 40 \
      --patience 2 \
      --milestones 8,14,20 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size 128 \
      --accu-steps 1 \
      --input-dim 40 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}/${loss}_emsize${embedding_size} \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}/${loss}_emsize${embedding_size}/checkpoint_40.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.3 \
      --s 15 \
      --remove-vad \
      --log-interval 10
  done
fi

if [ $stage -le 108 ]; then
  model=DTDNN
  datasets=vox2
  feat=log
  feat_type=spect
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_norm=Mean

  for model in ETDNN_v5; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_v2 \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat}_v2 \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --fix-length \
      --input-norm ${input_norm} \
      --nj 12 \
      --epochs 50 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size 128 \
      --accu-steps 1 \
      --input-dim 161 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}_v2/${loss}_100ce/emsize${embedding_size}_input${input_norm} \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_${encod}_v2/${loss}_100ce/emsize${embedding_size}_input${input_norm}/checkpoint_4.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.25 \
      --s 30 \
      --log-interval 10
  done
fi

if [ $stage -le 109 ]; then
  model=RET
  datasets=vox2
  feat=log
  feat_type=klsp
  loss=arcsoft
  encod=STAP
  embedding_size=512
  block_type=basic_v2
  input_norm=Mean
  batch_size=128
  resnet_size=14
  activation=leakyrelu
  scheduler=exp
  optimizer=sgd
  #  --dilation 1,2,3,1 \

  for encod in STAP ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --input-norm ${input_norm} \
      --nj 12 \
      --epochs 60 \
      --patience 2 \
      --random-chunk 200 400 \
      --milestones 10,20,30,40,50 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --block-type ${block_type} \
      --activation ${activation} \
      --weight-decay 0.00005 \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.000006 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size ${batch_size} \
      --accu-steps 1 \
      --input-dim 161 \
      --channels 512,512,512,512,512,1536 \
      --context 5,3,3,5 \
      --dilation 1,1,1,1 \
      --stride 1,1,1,1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}_${optimizer}_${scheduler}/input${input_norm}_em${embedding_size}_${block_type}_${activation}_wd5e5_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}_${optimizer}_${scheduler}/input${input_norm}_em${embedding_size}_${block_type}_${activation}_wd5e5_var/checkpoint_45.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --all-iteraion 0 \
      --log-interval 10
  done

  #  resnet_size=17
  #  for block_type in Basic; do
  #    echo -e "\n\033[1;4;31m stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
  #    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
  #    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
  #      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_v2 \
  #      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
  #      --train-trials trials_2w \
  #      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat}_v2 \
  #      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
  #      --fix-length \
  #      --input-norm ${input_norm} \
  #      --nj 12 \
  #      --epochs 60 \
  #      --patience 3 \
  #      --milestones 10,20,30,40,50 \
  #      --model ${model} \
  #      --resnet-size ${resnet_size} \
  #      --block-type ${block_type} \
  #      --scheduler rop \
  #      --weight-decay 0.00001 \
  #      --lr 0.1 \
  #      --alpha 0 \
  #      --feat-format kaldi \
  #      --embedding-size ${embedding_size} \
  #      --batch-size ${batch_size} \
  #      --accu-steps 1 \
  #      --input-dim 161 \
  #      --channels 512,512,512,512,512,1536 \
  #      --context 5,3,3,5 \
  #      --encoder-type ${encod} \
  #      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size} \
  #      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size}/checkpoint_21.pth \
  #      --cos-sim \
  #      --dropout-p 0.0 \
  #      --veri-pairs 9600 \
  #      --gpu-id 0,1 \
  #      --num-valid 2 \
  #      --loss-type ${loss} \
  #      --margin 0.25 \
  #      --s 30 \
  #      --all-iteraion 0 \
  #      --log-interval 10
  #  done
  exit
fi

if [ $stage -le 110 ]; then
  model=RET_v2
  datasets=vox2
  feat=log
  feat_type=klsp
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_norm=Mean
  batch_size=128
  resnet_size=18
  activation=relu
  #  --dilation 1,2,3,1 \

  for block_type in basic_v2 ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --input-norm ${input_norm} \
      --shuffle \
      --nj 12 \
      --epochs 55 \
      --patience 2 \
      --random-chunk 200 400 \
      --milestones 10,20,30,40,50 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --activation ${activation} \
      --block-type ${block_type} \
      --scheduler rop \
      --weight-decay 0.00001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size ${batch_size} \
      --accu-steps 1 \
      --input-dim 161 \
      --channels 256,256,512,512,1024,1024 \
      --context 5,3,3,3 \
      --stride 2,1,2,1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}/em${embedding_size}_input${input_norm}_${block_type}_${activation}_wde5_stride2121_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}/em${embedding_size}_input${input_norm}_${block_type}_${activation}_wde5_stride2121_var/checkpoint_5.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --all-iteraion 0 \
      --log-interval 10
  done
  exit
fi

if [ $stage -le 111 ]; then
  model=RET_v2
  datasets=vox2
  feat=log
  feat_type=spect
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_norm=Mean
  batch_size=128
  resnet_size=17
  stride=2

  for block_type in Basic; do
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_v2 \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat}_v2 \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
      --input-norm ${input_norm} \
      --nj 12 \
      --epochs 50 \
      --patience 3 \
      --milestones 10,20,30 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --block-type ${block_type} \
      --scheduler rop \
      --weight-decay 0.00001 \
      --lr 0.01 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size ${batch_size} \
      --accu-steps 1 \
      --input-dim 161 \
      --channels 512,512,512,512,512,1536 \
      --context 5,3,3,5 \
      --stride 1,${stride},1,${stride} \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size}_stride1${stride}_wde5_shuf \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size}_stride1${stride}_wde5_shuf/checkpoint_57.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.25 \
      --s 30 \
      --all-iteraion 0 \
      --log-interval 10
  done

#  resnet_size=17
#  for block_type in Basic ; do
#    echo -e "\n\033[1;4;31m stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
#    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
#    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_v2 \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_${feat}/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat}_v2 \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_${feat} \
#      --fix-length \
#      --input-norm ${input_norm} \
#      --nj 12 \
#      --epochs 60 \
#      --patience 3 \
#      --milestones 10,20,30,40,50 \
#      --model ${model} \
#      --resnet-size ${resnet_size} \
#      --block-type ${block_type} \
#      --scheduler rop \
#      --weight-decay 0.00001 \
#      --lr 0.1 \
#      --alpha 0 \
#      --feat-format kaldi \
#      --embedding-size ${embedding_size} \
#      --batch-size ${batch_size} \
#      --accu-steps 1 \
#      --input-dim 161 \
#      --channels 512,512,512,512,512,1536 \
#      --context 5,3,3,5 \
#      --encoder-type ${encod} \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size} \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_v2/${loss}_0ce/em${embedding_size}_input${input_norm}_${block_type}_bs${batch_size}/checkpoint_21.pth \
#      --cos-sim \
#      --dropout-p 0.0 \
#      --veri-pairs 9600 \
#      --gpu-id 0,1 \
#      --num-valid 2 \
#      --loss-type ${loss} \
#      --margin 0.25 \
#      --s 30 \
#      --all-iteraion 0 \
#      --log-interval 10
#  done
fi


if [ $stage -le 112 ]; then
  model=RET_v3
  datasets=vox2
  feat=log
  feat_type=klfb
  loss=arcsoft
  encod=STAP
  embedding_size=256
  block_type=shublock
  input_norm=Mean
  batch_size=128
  resnet_size=18
  activation=leakyrelu
  #  --dilation 1,2,3,1 \

  for encod in SASP2 ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb40 \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb40/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb40_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb40 \
      --input-norm ${input_norm} \
      --shuffle \
      --nj 12 \
      --epochs 60 \
      --patience 3 \
      --random-chunk 200 200 \
      --milestones 10,20,30,40,50 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --block-type ${block_type} \
      --activation ${activation} \
      --scheduler rop \
      --weight-decay 0.00001 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --batch-size ${batch_size} \
      --accu-steps 1 \
      --input-dim 40 \
      --channels 512,512,512,512,512,1536 \
      --context 5,3,3,3 \
      --dilation 1,2,3,4 \
      --stride 1,1,1,1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}/${input_norm}_em${embedding_size}_${block_type}_${activation}_dila4_wde5_2s \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_${encod}_baseline/${loss}/${input_norm}_em${embedding_size}_${block_type}_${activation}_dila4_wde5_2s/checkpoint_10.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --remove-vad \
      --all-iteraion 0 \
      --log-interval 10
  done
fi


# CnCeleb
if [ $stage -le 150 ]; then
  model=TDNN_v5
  datasets=cnceleb
  #  feat=fb24
#  feat_type=pyfb
  feat_type=klfb
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_dim=40
  input_norm=Mean
  lr_ratio=0
  loss_ratio=10
  subset=
  activation=leakyrelu
  scheduler=cyclic
  optimizer=adam
  stat_type=margin1 #margin1sum
  m=1.0

  # _lrr${lr_ratio}_lsr${loss_ratio}

 for stat_type in margin1 ; do
   feat=fb${input_dim}
   #_ws25
   if [ "$loss" == "arcdist" ]; then
     loss_str=_lossr${loss_ratio}_${stat_type}${m}_lambda
   elif [ "$loss" == "arcsoft" ]; then
     loss_str=
   fi
   echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
#    kernprof -l -v TrainAndTest/train_egs.py \
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_distributed.py \
     --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_${feat} \
     --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_${feat}_valid \
     --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
     --nj 12 \
     --shuffle \
     --epochs 60 \
     --patience 3 \
     --milestones 10,20,30,40 \
     --model ${model} \
     --optimizer ${optimizer} \
     --scheduler ${scheduler} \
     --weight-decay 0.0005 \
     --lr 0.001 \
     --base-lr 0.00000001 \
     --alpha 0 \
     --feat-format kaldi \
     --embedding-size ${embedding_size} \
     --batch-size 128 \
     --random-chunk 200 400 \
     --input-dim ${input_dim} \
     --activation ${activation} \
     --channels 512,512,512,512,1500 \
     --encoder-type ${encod} \
     --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs${subset}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${encod}_em${embedding_size}${loss_str}_wd5e4_var \
     --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs${subset}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${encod}_em${embedding_size}${loss_str}_wd5e4_var/checkpoint_40.pth \
     --cos-sim \
     --dropout-p 0.0 \
     --veri-pairs 9600 \
     --gpu-id 2 \
     --num-valid 2 \
     --loss-ratio ${loss_ratio} \
     --lr-ratio ${lr_ratio} \
     --loss-lambda \
     --loss-type ${loss} \
     --margin 0.2 \
     --m ${m} \
     --s 30 \
     --remove-vad \
     --stat-type $stat_type \
     --log-interval 10
  done
  exit
fi


if [ $stage -le 151 ]; then

  model=TDNN_v5
  datasets=cnceleb
  feat_type=klfb
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_dim=40
  input_norm=Mean
  lr_ratio=0
  loss_ratio=1
  # _lrr${lr_ratio}_lsr${loss_ratio}

 for loss in arcdist; do
   feat=fb${input_dim}
   python -W ignore TrainAndTest/train_egs.py \
     --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
     --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
     --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
     --nj 12 \
     --shuffle \
     --epochs 29 \
     --patience 3 \
     --milestones 10,20,30,40 \
     --model ${model} \
     --optimizer ${optimizer} \
     --scheduler ${scheduler} \
     --weight-decay 0.0005 \
     --lr 0.01 \
     --alpha 0 \
     --feat-format kaldi \
     --embedding-size ${embedding_size} \
     --batch-size 128 \
     --random-chunk 200 400 \
     --input-dim ${input_dim} \
     --channels 512,512,512,512,1500 \
     --encoder-type ${encod} \
     --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_lr${loss_ratio}_wd5e4_var \
     --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_lr${loss_ratio}_wd5e4_var/checkpoint_21.pth \
     --cos-sim \
     --dropout-p 0.0 \
     --veri-pairs 9600 \
     --gpu-id 0,1 \
     --num-valid 2 \
     --loss-ratio ${loss_ratio} \
     --lr-ratio ${lr_ratio} \
     --loss-type ${loss} \
     --margin 0.2 \
     --s 30 \
     --remove-vad \
     --log-interval 10
 done
fi

if [ $stage -le 155 ]; then
  model=TDNN_v5
  datasets=cnceleb
  embedding_size=512
  block_type=basic
  loss=arcsoft
  scheduler=rop
  optimizer=sgd
  activation=leakyrelu

  # num_centers=3
  dev_sub=12

  for loss in arcsoft; do
    feat=fb${input_dim}
  #   #_ws25
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
   python -W ignore TrainAndTest/train_egs.py \
     --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${dev_sub}_${feat} \
     --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${dev_sub}_${feat}_valid \
     --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
     --nj 12 \
     --shuffle \
     --epochs 60 \
     --patience 3 \
     --milestones 10,20,30,40 \
     --model ${model} \
     --optimizer ${optimizer} \
     --scheduler ${scheduler} \
     --weight-decay 0.0005 \
     --lr 0.1 \
     --alpha 0 \
     --feat-format kaldi \
     --embedding-size ${embedding_size} \
     --batch-size 128 \
     --random-chunk 200 400 \
     --input-dim ${input_dim} \
     --activation ${activation} \
     --channels 512,512,512,512,1500 \
     --encoder-type ${encod} \
     --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs${dev_sub}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${encod}_em${embedding_size}_${activation}_wd5e4_var \
     --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs${dev_sub}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${encod}_em${embedding_size}_${activation}_wd5e4_var/checkpoint_40.pth \
     --cos-sim \
     --dropout-p 0.0 \
     --veri-pairs 9600 \
     --gpu-id 0,1 \
     --num-valid 2 \
     --loss-ratio ${loss_ratio} \
     --lr-ratio ${lr_ratio} \
     --loss-type ${loss} \
     --margin 0.2 \
     --s 30 \
     --remove-vad \
     --log-interval 10
  done
  exit
fi


if [ $stage -le 156 ]; then
    model=TDNN_v5
    datasets=cnceleb

   python -W ignore TrainAndTest/train_egs.py \
     --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
     --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
     --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
     --nj 12 \
     --shuffle \
     --epochs 42 \
     --patience 3 \
     --milestones 10,20,30,40 \
     --model ${model} \
     --scheduler rop \
     --weight-decay 0.0005 \
     --lr 0.1 \
     --alpha 0 \
     --feat-format kaldi \
     --embedding-size ${embedding_size} \
     --batch-size 128 \
     --random-chunk 200 400 \
     --input-dim ${input_dim} \
     --channels 512,512,512,512,1500 \
     --encoder-type ${encod} \
     --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var \
     --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_10.pth \
     --cos-sim \
     --dropout-p 0.0 \
     --veri-pairs 9600 \
     --gpu-id 0,1 \
     --num-valid 2 \
     --loss-ratio ${loss_ratio} \
     --lr-ratio ${lr_ratio} \
     --loss-type ${loss} \
     --margin 0.2 \
     --s 30 \
     --remove-vad \
     --log-interval 10
fi

if [ $stage -le 157 ]; then
  model=TDNN_v5
  datasets=cnceleb
  embedding_size=512
  encod=STAP
  block_type=basic
  input_norm=Mean
  loss=arcsoft
  scheduler=rop
  optimizer=sgd
  input_dim=40
  lr_ratio=0
  loss_ratio=0
  feat_type=klfb

  num_centers=3
  dev_sub=
  batch_size=256
  mask_layer=baseline
  # _center${num_centers}

  for loss in arcsoft ; do
    feat=fb${input_dim}
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${dev_sub}_${feat} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${dev_sub}_${feat}_valid \
      --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
      --nj 12 \
      --shuffle \
      --epochs 50 \
      --patience 3 \
      --milestones 10,20,30,40 \
      --model ${model} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --base-lr 0.00001 \
      --alpha 0 \
      --feat-format kaldi \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --batch-size ${batch_size} \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs${dev_sub}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_${encod}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs${dev_sub}_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_17.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-ratio ${loss_ratio} \
      --lr-ratio ${lr_ratio} \
      --loss-type ${loss} \
      --num-center ${num_centers} \
      --margin 0.2 \
      --s 30 \
      --remove-vad \
      --log-interval 10 \
      --test-interval 2
  done
  # exit
fi

if [ $stage -le 158 ]; then
  model=TDNN_v5
  datasets=cnceleb
  embedding_size=512
  encod=STAP
  block_type=basic
  input_norm=Mean
  mask_layer=drop
  input_dim=40
  lr_ratio=0
  loss_ratio=0
  feat_type=klfb

  weight=vox2_cf
  loss=arcsoft
  scheduler=rop
  optimizer=sgd
  feat=fb${input_dim}
  batch_size=256

  # for weight in vox2_cf; do
    #_ws25
  echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
  #   python -W ignore TrainAndTest/train_egs.py \
  #    --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
  #    --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
  #    --train-trials trials_2w \
  #    --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
  #    --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
  #    --nj 12 \
  #    --epochs 50 \
  #    --batch-size ${batch_size} \
  #    --patience 3 \
  #    --milestones 10,20,30,40 \
  #    --model ${model} \
  #    --optimizer ${optimizer} \
  #    --scheduler ${scheduler} \
  #    --weight-decay 0.0005 \
  #    --lr 0.1 \
  #    --base-lr 0.00001 \
  #    --alpha 0 \
  #    --feat-format kaldi \
  #    --embedding-size ${embedding_size} \
  #    --random-chunk 200 400 \
  #    --input-dim ${input_dim} \
  #    --channels 512,512,512,512,1500 \
  #    --mask-layer ${mask_layer} \
  #    --encoder-type ${encod} \
  #    --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_wd5e4_var \
  #    --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_20.pth \
  #    --cos-sim \
  #    --dropout-p 0.0 \
  #    --veri-pairs 9600 \
  #    --gpu-id 0,1 \
  #    --num-valid 2 \
  #    --loss-ratio ${loss_ratio} \
  #    --lr-ratio ${lr_ratio} \
  #    --loss-type ${loss} \
  #    --margin 0.2 \
  #    --s 30 \
  #    --remove-vad \
  #    --log-interval 10 \
  #    --test-interval 2
  # done

  mask_layer=drop
  weight_p=0
  scale=0.2
  weight=vox2_cf
  for scale in 0.2 0.3; do
    python -W ignore TrainAndTest/train_egs.py \
     --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
     --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
     --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
     --nj 12 \
     --epochs 50 \
     --batch-size ${batch_size} \
     --input-norm ${input_norm} \
     --patience 3 \
     --milestones 10,20,30,40 \
     --model ${model} \
     --optimizer ${optimizer} \
     --scheduler ${scheduler} \
     --weight-decay 0.0005 \
     --lr 0.1 \
     --base-lr 0.00001 \
     --alpha 0 \
     --feat-format kaldi \
     --embedding-size ${embedding_size} \
     --random-chunk 200 400 \
     --input-dim ${input_dim} \
     --channels 512,512,512,512,1500 \
     --mask-layer ${mask_layer} \
     --init-weight ${weight} \
     --weight-p ${weight_p} \
     --scale ${scale} \
     --encoder-type ${encod} \
     --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_${weight}scale${scale}p${weight_p}_wd5e4_var \
     --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_${weight}scale${scale}p${weight_p}_wd5e4_var/checkpoint_20.pth \
     --cos-sim \
     --dropout-p 0.0 \
     --veri-pairs 9600 \
     --gpu-id 0,1 \
     --num-valid 2 \
     --loss-ratio ${loss_ratio} \
     --lr-ratio ${lr_ratio} \
     --loss-type ${loss} \
     --margin 0.2 \
     --s 30 \
     --remove-vad \
     --log-interval 10 \
     --test-interval 2
done
exit
fi


# Aishell2
if [ $stage -le 200 ]; then
  #  model=TDNN
  # datasets=aidata
  datasets=aishell2

  feat=fb40
  feat_type=klfb
  loss=arcsoft
  encod=STAP
  embedding_size=512
  input_norm=Mean
  batch_size=256
  scheduler=rop
  optimizer=sgd

  for model in TDNN_v5; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # kernprof -l -v TrainAndTest/Spectrogram/train_egs.py \
    python -W ignore TrainAndTest/train_egs.py \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat}_valid \
      --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
      --input-dim 40 --input-norm ${input_norm} \
      --random-chunk 200 400 --nj 12 \
      --epochs 50 --batch-size ${batch_size} --patience 3 \
      --milestones 10,20,30,40 \
      --model ${model} \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --channels 512,512,512,512,1500 \
      --embedding-size ${embedding_size} \
      --accu-steps 1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_53.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --all-iteraion 0 \
      --remove-vad \
      --log-interval 10
  done
  exit
fi


if [ $stage -le 300 ]; then
  model=ECAPA
  datasets=vox2
  #  feat=fb24
#  feat_type=pyfb
  feat_type=klfb
  loss=arcsoft
  encod=SASP2 embedding_size=192
  input_norm=Mean input_dim=40
  lr_ratio=0
  loss_ratio=10
  subset=
  activation=leakyrelu
  scheduler=cyclic optimizer=adam
  stat_type=margin1 #margin1sum
  m=1.0

  # _lrr${lr_ratio}_lsr${loss_ratio}

 for seed in 123456 123457 123458 ; do
   feat=fb${input_dim}

   echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
#   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist.py
  # CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/TDNNs/vox2_ecapa.yaml --seed=${seed}

  #  CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=417420 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Fbank/TDNNs/vox2_ecapa_mixup.yaml --lamda-beta 0.2 --seed=${seed}
  CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=417420 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/TDNNs/vox1_tdnn.yaml --seed=${seed}
  CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=417420 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/TDNNs/vox1_tdnn_drop.yaml --seed=${seed}

#   CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Fbank/TDNNs/vox2_tdnn_mixup.yaml --seed=${seed}
  done
  exit
fi


if [ $stage -le 301 ]; then
  datasets=vox2 testset=vox1
  model=TDNN_v5 resnet_size=34
  encoder_type=STAP
  alpha=0
  block_type=res2tdnn
  embedding_size=512
  input_norm=Mean
  loss=arcsoft
  feat_type=klfb
  sname=dev

  mask_layer=baseline
  scheduler=rop optimizer=sgd
  input_dim=40
  batch_size=256
  chn=512
#  fast=none1
#  downsample=k5
  for seed in 123456 123457 123458 ; do
  for sname in dev_fb40 ; do
    if [ $chn -eq 512 ]; then
      channels=512,512,512,512,1536
      chn_str=
    elif [ $chn -eq 1024 ]; then
      channels=1024,1024,1024,1024,3072
      chn_str=chn1024_
    fi
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model} in ${datasets}_egs with ${loss} \033[0m\n"
    model_dir=${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_${encoder_type}_em${embedding_size}_${chn_str}wd2e5_vares_bashuf2/${seed}
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb40 \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb40 \
      --feat-format kaldi \
      --shuffle --batch-shuffle \
      --input-norm ${input_norm} --input-dim ${input_dim} --remove-vad \
      --batch-size ${batch_size} \
      --nj 6 --epochs 80 \
      --random-chunk 200 400 \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --patience 2 \
      --early-stopping --early-patience 20 --early-delta 0.01 --early-meta EER \
      --cyclic-epoch 4 \
      --accu-steps 1 \
      --lr 0.1 --base-lr 0.000001 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_21.pth \
      --channels ${channels} \
      --embedding-size ${embedding_size} --encoder-type ${encoder_type} \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --grad-clip 0 \
      --lr-ratio 0.01 \
      --weight-decay 0.00002 \
      --gpu-id 0,3 \
      --extract \
      --cos-sim
  done
  done
  exit
fi


