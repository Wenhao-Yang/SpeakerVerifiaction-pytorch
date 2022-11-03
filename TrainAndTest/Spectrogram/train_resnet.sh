#!/usr/bin/env bash

stage=60

waited=0
while [ $(ps 10392 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  for loss in soft asoft amsoft center; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet.py \
      --model ResNet \
      --resnet-size 18 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 512 \
      --batch-size 128 \
      --test-batch-size 32 \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --lr 0.1 \
      --margin 0.35 \
      --s 30 \
      --m 4 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet/18/spect/${loss} \
      --resume Data/checkpoint/ResNet/18/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi

if [ $stage -le 20 ]; then
  datasets=vox1 testset=vox1
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  alpha=0
  block_type=cbam_v2
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev

  mask_layer=baseline
  scheduler=rop optimizer=sgd
  fast=none1 downsample=k5
  sname=dev
  subset=

  for subset in female ; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_rvec_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp02_alpha${alpha}_${fast}_wd5e4_var_es
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_${subset} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_${subset}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --nj 12 --epochs 50 \
      --random-chunk 200 400 \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --patience 3 \
      --early-stopping --early-patience 15 --early-delta 0.01 \
      --early-meta MinDCF_01 \
      --mask-layer ${mask_layer} --init-weight v1_f2m \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_10.pth \
      --channels 16,32,64,128 \
      --downsample ${downsample} --fast ${fast} \
      --input-dim 161 \
      --block-type ${block_type} \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.2 \
      --gpu-id 0,1 \
      --shuffle \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
#      --grad-clip 5 \

#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
#      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/${sname}/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 60 \
#      --random-chunk 200 400 \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --patience 3 \
#      --accu-steps 1 \
#      --lr 0.1 \
#      --milestones 10,20,40,50 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wd5e4_var_${sname} \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wd5e4_var_${sname}/checkpoint_10.pth \
#      --channels 32,64,128,256 \
#      --downsample ${downsample} \
#      --input-dim 161 \
#      --fast ${fast} \
#      --block-type ${block_type} \
#      --stride 2 \
#      --batch-size 128 \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --grad-clip 0 \
#      --s 30 \
#      --lr-ratio 0.01 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --shuffle \
#      --all-iteraion 0 \
#      --extract \
#      --cos-sim \
#      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 21 ]; then
  datasets=vox1
  model=ThinResNet resnet_size=34
  encoder_type=AVG
  alpha=0
  block_type=basic_v2
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
  downsample=k3
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
    for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --resnet-size ${resnet_size} \
      --downsample ${downsample} \
      --nj 12 \
      --epochs 50 \
      --optimizer sgd \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --fast none1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var/checkpoint_25.pth \
      --kernel-size 5,5 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} \
      --red-ratio 8 \
      --stride 2,2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --shuffle \
      --cos-sim \
      --loss-type ${loss}
  done

  for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --downsample ${downsample} \
      --nj 12 --epochs 50 \
      --optimizer sgd --scheduler rop \
      --patience 2 \
      --mask-layer attention --init-weight vox2 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var/checkpoint_25.pth \
      --kernel-size 5,5 --stride 2,2 --fast none1 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} --red-ratio 8 \
      --batch-size 128 \
      --time-dim 1 --avg-size 5 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --shuffle \
      --all-iteraion 0 \
      --extract --cos-sim
  done

#  for sname in dev; do
#  for block_type in basic seblock cbam; do
#  for block_type in basic_v2 ; do
#    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} --resnet-size ${resnet_size} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --mask-layer attention --init-weight vox2 \
#      --random-chunk 200 400 \
#      --nj 12 \
#      --epochs 50 \
#      --optimizer sgd --scheduler rop \
#      --patience 2 \
#      --accu-steps 1 \
#      --lr 0.1 \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var/checkpoint_10.pth \
#      --kernel-size 5,5 --stride 2 \
#      --channels 16,32,64,128 \
#      --input-dim 161 \
#      --block-type ${block_type} --red-ratio 8 \
#      --batch-size 128 \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 --s 30 \
#      --grad-clip 0 \
#      --lr-ratio 0.01 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.25 \
#      --gpu-id 0,1 \
#      --all-iteraion 0 \
#      --extract --cos-sim \
#      --loss-type ${loss}
#  done
  exit
fi

if [ $stage -le 22 ]; then
  datasets=vox2
  model=LoResNet resnet_size=8
  encoder_type=None
  alpha=0
  block_type=cbam
  embedding_size=512
  input_norm=Mean
  loss=arcsoft
  dropout_p=0.1
  feat_type=klsp
  sname=dev

  for sname in dev; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp01_wde4_${sname}_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp01_wde4_${sname}_var/checkpoint_5.pth \
      --channels 64,128,256 \
      --input-dim 161 \
      --block-type ${block_type} \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size 4 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p ${dropout_p} \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 50 ]; then
  datasets=vox1 testsets=vox1
  model=ThinResNet resnet_size=50
  encoder_type=SAP2
  alpha=0
  block_type=basic
  batch_size=256
  chn=16
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
#  downsample=k3
  downsample=k1

  mask_layer=rvec weight=rclean
  scheduler=rop optimizer=sgd
  fast=none1
  dropout_p=0.1
  avg_size=4
  weight_p=0 scale=0.2
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
  for resnet_size in 34 18; do
    for seed in 123456 ;do
    if [ $resnet_size -le 34 ];then
      expansion=1
      batch_size=256
    else
      expansion=2
      batch_size=128
    fi

    chn_str=
    channels=16,32,64,128

    if [ $chn -eq 32 ]; then
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

    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wde4_var/${seed}

    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
      --feat-format kaldi --seed ${seed} \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --nj 6 --epochs 60 \
      --patience 3 --early-stopping --early-patience 15 --early-delta 0.01 --early-meta EER \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --mask-layer ${mask_layer} --init-weight ${weight} \
      --kernel-size 5,5 --stride 2,2 --fast ${fast} \
      --expansion ${expansion} \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} --red-ratio 8 --downsample ${downsample} \
      --batch-size ${batch_size} \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size ${avg_size} \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_25.pth \
      --loss-type ${loss} --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p ${dropout_p} \
      --gpu-id 0,2 \
      --all-iteraion 0 \
      --extract --shuffle --cos-sim
  done
  done
  exit

fi


if [ $stage -le 60 ]; then
  model=ThinResNet
  datasets=vox1
  feat_type=klsp
  loss=arcsoft
  encod=SAP2
  embedding_size=256
  input_dim=40 input_norm=Mean
  lr_ratio=0 loss_ratio=10
  subset=
  activation=leakyrelu
  scheduler=rop optimizer=sgd
  stat_type=margin1 #margin1sum
  m=1.0

  # _lrr${lr_ratio}_lsr${loss_ratio}
 for seed in 123456 ; do
   echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Spectrogram/ResNets/vox2_resnet.yaml --seed=${seed}

#    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Fbank/ResNets/aidata_resnet_mixup.yaml --seed=${seed}
  done
  exit
fi