#!/usr/bin/env bash

stage=10
waited=0
while [ $(ps 27253 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

#if [ $stage -le 0 ]; then
#
#
#fi
if [ $stage -le 0 ]; then
  datasets=vox1
  feat_type=klsp
  model=LoResNet
  resnet_size=8
  encoder_type=AVG
  embedding_size=256
  block_type=cbam
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  nj=8
  weight=clean

  teacher_dir=Data/checkpoint/LoResNet8/vox1/klsp_egs_baseline/arcsoft_sgd_rop/Mean_cbam_AVG_dp25_alpha0_em256_wd5e4_var
  label_dir=Data/label/LoResNet8/vox1/klsp_egs_baseline/arcsoft_sgd_rop/Mean_cbam_AVG_dp25_alpha0_em256_wd5e4_var
  kd_type=vanilla_attention
#  _${weight}

   for encoder_type in AVG ; do
     echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_kd.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
       --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev/trials_dir \
       --train-trials trials_2w \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
       --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test \
       --feat-format kaldi \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj ${nj} \
       --shuffle \
       --epochs 50 \
       --patience 3 \
       --batch-size 128 \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --init-weight ${weight} \
       --milestones 10,20,30,40 \
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var_${kd_type} \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var_${kd_type}/checkpoint_50.pth \
       --kernel-size ${kernel} \
       --channels 16,32,64 \
       --stride 2 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size 4 \
       --encoder-type ${encoder_type} \
       --num-valid 2 \
       --alpha ${alpha} \
       --margin 0.2 \
       --s 30 \
       --weight-decay 0.0005 \
       --dropout-p 0.125 \
       --gpu-id 0,1 \
       --extract \
       --cos-sim \
       --all-iteraion 0 \
       --loss-type ${loss} \
       --kd-type ${kd_type} \
       --kd-loss kld \
       --distil-weight 0.5 \
       --teacher-model-yaml ${teacher_dir}/model.2022.02.23.yaml \
       --teacher-resume ${teacher_dir}/checkpoint_40.pth \
       --temperature 20
   done

#  weight=vox2
#  scale=0.2
#
#  for mask_layer in drop; do
##    mask_layer=baseline
#    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs_${mask_layer} with ${loss} with ${input_norm} normalization \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj ${nj} \
#      --epochs 50 \
#      --batch-size 128 \
#      --shuffle \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.000005 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --scale ${scale} \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var/checkpoint_50.pth \
#      --kernel-size ${kernel} \
#      --channels 64,128,256 \
#      --stride 2 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 4 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --loss-type ${loss}
#  done

  exit
fi

if [ $stage -le 5 ]; then
  datasets=vox1
  feat_type=klsp
  model=LoResNet
  resnet_size=8
  encoder_type=AVG
  embedding_size=256
  block_type=cbam
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=attention
  scheduler=rop
  optimizer=sgd
  nj=8
  weight=vox2

  teacher_dir=Data/checkpoint/LoResNet8/vox1/klsp_egs_baseline/arcsoft/None_cbam_em256_alpha0_dp25_wd5e4_dev_var

   for encoder_type in AVG ; do
     echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_kd.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
       --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev/trials_dir \
       --train-trials trials_2w \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
       --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test \
       --feat-format kaldi \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj ${nj} \
       --shuffle \
       --epochs 50 \
       --batch-size 128 \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --init-weight ${weight} \
       --milestones 10,20,30,40 \
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_chn32_var \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_chn32_var/checkpoint_50.pth \
       --kernel-size ${kernel} \
       --channels 32,64,128 \
       --stride 2 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size 4 \
       --encoder-type ${encoder_type} \
       --num-valid 2 \
       --alpha ${alpha} \
       --margin 0.2 \
       --s 30 \
       --weight-decay 0.0005 \
       --dropout-p 0.2 \
       --gpu-id 0,1 \
       --extract \
       --cos-sim \
       --all-iteraion 0 \
       --loss-type ${loss} \
       --distil-weight 0.5 \
       --teacher-model-yaml ${teacher_dir}/model.2022.01.05.yaml \
       --teacher-resume ${teacher_dir}/checkpoint_40.pth \
       --temperature 20
   done

#  weight=vox2
#  scale=0.2
#
#  for mask_layer in drop; do
##    mask_layer=baseline
#    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs_${mask_layer} with ${loss} with ${input_norm} normalization \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj ${nj} \
#      --epochs 50 \
#      --batch-size 128 \
#      --shuffle \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.000005 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --scale ${scale} \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var/checkpoint_50.pth \
#      --kernel-size ${kernel} \
#      --channels 64,128,256 \
#      --stride 2 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 4 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --loss-type ${loss}
#  done

  exit
fi


if [ $stage -le 10 ]; then
  datasets=vox1
  feat_type=klsp
  model=ThinResNet
  resnet_size=8
  encoder_type=AVG
  embedding_size=256
  block_type=basic
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=rvec
  scheduler=rop
  optimizer=sgd
  nj=8
  weight=clean
  avg_size=5
  dp=0.1

  teacher_dir=Data/checkpoint/ThinResNet34/vox1/klsp_egs_rvec/123458/arcsoft_sgd_rop/Mean_batch256_basic_downk1_avg5_SAP2_em256_dp01_alpha0_none1_wde4_var
  kd_type=embedding_cos #em_l2 vanilla
  kd_ratio=0.4
  kd_loss=
  chn=16
#  _${weight}
  for chn in 16 ; do
  for seed in 123456 1234567 123458; do

     if [ $resnet_size -le 34 ];then
        expansion=1
        batch_size=256
      else
        expansion=4
        batch_size=256
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
      if [[ $mask_layer == attention* ]];then
        at_str=_${weight}
        if [[ $weight_norm != max ]];then
          at_str=${at_str}${weight_norm}
        fi
      elif [ "$mask_layer" = "drop" ];then
        at_str=_${weight}_dp${weight_p}s${scale}
      elif [ "$mask_layer" = "both" ];then
        at_str=_`echo $mask_len | sed  's/,//g'`
      else
        at_str=
      fi

    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wde4_var_${kd_type}${kd_ratio}${kd_loss}/${seed}

#    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_kd_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp${dp_str}_alpha${alpha}_em${embedding_size}_wd5e4_chn${chn}_var_${kd_type}${kd_ratio}${kd_loss}
#           --kd-loss ${kd_loss} \

     echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_kd.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
       --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev/trials_dir \
       --train-trials trials_2w \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
       --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test \
       --feat-format kaldi \
       --seed $seed \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj ${nj} \
       --shuffle \
       --epochs 50 \
       --batch-size 128 \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --init-weight ${weight} \
       --milestones 10,20,30,40 \
       --check-path Data/checkpoint/${model_dir} \
       --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
       --kernel-size ${kernel} \
       --channels ${channels} \
       --stride 2 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size ${avg_size} \
       --encoder-type ${encoder_type} \
       --num-valid 2 \
       --alpha ${alpha} \
       --margin 0.2 \
       --s 30 \
       --weight-decay 0.0005 \
       --dropout-p ${dp} \
       --gpu-id 1 \
       --extract \
       --cos-sim \
       --all-iteraion 0 \
       --loss-type ${loss} \
       --kd-type ${kd_type} \
       --distil-weight 0.5 \
       --kd-ratio ${kd_ratio} \
       --teacher-model-yaml ${teacher_dir}/model.2022.07.01.yaml \
       --teacher-resume ${teacher_dir}/best.pth \
       --temperature 20
   done
   done
fi