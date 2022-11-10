#!/usr/bin/env bash

stage=60

waited=0
while [ $(ps 182247 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  for loss in soft asoft amsoft center; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet.py \
      --model ResNet --resnet-size 18 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 512 \
      --batch-size 128 --test-batch-size 32 \
      --nj 12 --epochs 24 \
      --milestones 10,15,20 \
      --lr 0.1 \
      --loss-type ${loss} --margin 0.35 --s 30 --m 4 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet/18/spect/${loss} \
      --resume Data/checkpoint/ResNet/18/spect/${loss}/checkpoint_1.pth

  done
fi

if [ $stage -le 20 ]; then
  datasets=vox1
  model=ThinResNet resnet_size=34
  encoder_type=STAP embedding_size=256
  alpha=0
  block_type=basic
  input_norm=Mean
  loss=soft
  feat_type=klsp
  sname=dev

  for sname in dev dev_aug_com; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} --input-dim 161 \
      --nj 12 --epochs 60 --batch-size 128 \
      --scheduler rop --patience 3 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde3_${sname}_adam \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde3_${sname}_adam/checkpoint_10.pth \
      --channels 16,32,64,128 \
      --block-type ${block_type} \
      --stride 2 \
      --time-dim 1 --avg-size 4 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 --all-iteraion 0 \
      --lr-ratio 0.01 --weight-decay 0.001 \
      --dropout-p 0 \
      --gpu-id 0,1 --grad-clip 0 \
      --extract --cos-sim \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 21 ]; then
  datasets=vox1
  model=ThinResNet resnet_size=34
  encoder_type=AVG embedding_size=256
  alpha=0
  block_type=basic_v2
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
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} --input-dim 161 \
      --random-chunk 200 400 \
      --downsample ${downsample} \
      --nj 12 --epochs 50 --batch-size 128 \
      --optimizer sgd --scheduler rop \
      --patience 2 --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var/checkpoint_25.pth \
      --kernel-size 5,5 --stride 2,2 --fast none1 \
      --channels 16,32,64,128 \
      --block-type ${block_type} --red-ratio 8 \
      --time-dim 1 --avg-size 5 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract --shuffle --cos-sim \
      --loss-type ${loss}
  done

  for downsample in k5; do
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} --random-chunk 200 400 \
      --nj 12 --epochs 50 --batch-size 128 \
      --optimizer sgd --scheduler rop \
      --patience 2 \
      --mask-layer attention --init-weight vox2 \
      --accu-steps 1 \
      --fast none1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_25.pth \
      --kernel-size 5,5 --stride 2,2 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} --downsample ${downsample} --red-ratio 8 \
      --time-dim 1 --avg-size 5 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --grad-clip 0 \
      --lr-ratio 0.01 --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --shuffle --extract \
      --cos-sim
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
#      --mask-layer attention --init-weight vox2 --random-chunk 200 400 \
#      --nj 12 --epochs 50 \
#      --optimizer sgd --scheduler rop \
#      --patience 2 \
#      --accu-steps 1 \
#      --lr 0.1 \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var/checkpoint_10.pth \
#      --channels 16,32,64,128 \
#      --input-dim 161 \
#      --block-type ${block_type} --red-ratio 8 \
#      --kernel-size 5,5 --stride 2 \
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
  encoder_type=None embedding_size=512
  alpha=0
  block_type=cbam
  input_norm=Mean
  loss=arcsoft
  dropout_p=0.1
  feat_type=klsp
  sname=dev

  for sname in dev; do
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp01_wde4_${sname}_var
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
      --nj 12 --epochs 50 --batch-size 128 \
      --scheduler rop --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_5.pth \
      --channels 64,128,256 \
      --input-dim 161 \
      --block-type ${block_type} \
      --stride 2 \
      --time-dim 1 --avg-size 4 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 --s 30 \
      --grad-clip 0 \
      --lr-ratio 0.01 --weight-decay 0.0001 \
      --dropout-p ${dropout_p} \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract --cos-sim \
      --loss-type ${loss}
  done
  exit
fi


if [ $stage -le 40 ]; then
  datasets=vox1 testsets=vox1
  model=ThinResNet resnet_size=34
#  resnet_size=50
  encoder_type=SAP2
  alpha=0
  block_type=basic downsample=k1 expansion=4
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
  batch_size=256

  mask_layer=rvec
  scheduler=rop optimizer=sgd
  fast=none1
  chn=16
  cyclic_epoch=8
  avg_size=5
#  nesterov
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
  for seed in 123456 123457 123458 ;do
    for chn in 16 ; do
      if [ $resnet_size -le 34 ];then
        expansion=1
      else
        expansion=4
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

      echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"

      model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_${chn_str}wde4_var

      python TrainAndTest/train_egs.py \
        --model ${model} --resnet-size ${resnet_size} \
        --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
        --train-test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
        --train-trials trials \
        --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
        --test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
        --feat-format kaldi \
        --input-norm ${input_norm} --input-dim 161 \
        --seed ${seed} \
        --random-chunk 200 400 \
        --optimizer ${optimizer} --scheduler ${scheduler} \
        --cyclic-epoch ${cyclic_epoch} \
        --nj 12 --epochs 60 \
        --patience 3 --early-stopping --early-patience 15 --early-delta 0.0001 --early-meta EER \
        --accu-steps 1 \
        --fast ${fast} \
        --lr 0.1 --base-lr 0.000001 \
        --milestones 10,20,30,40 \
        --check-path Data/checkpoint/${model_dir} \
        --resume Data/checkpoint/${model_dir}/checkpoint_25.pth \
        --kernel-size 5,5 --stride 2,2 \
        --channels ${channels} \
        --block-type ${block_type} --downsample ${downsample} --expansion ${expansion} \
        --batch-size ${batch_size} \
        --embedding-size ${embedding_size} \
        --time-dim 1 --avg-size ${avg_size} \
        --encoder-type ${encoder_type} \
        --num-valid 2 \
        --alpha ${alpha} \
        --margin 0.2 --s 30 \
        --grad-clip 0 \
        --lr-ratio 0.01 \
        --weight-decay 0.0001 \
        --dropout-p 0.1 \
        --gpu-id 4,5 \
        --all-iteraion 0 \
        --extract --shuffle --cos-sim \
        --loss-type ${loss}
    done
  done
exit
fi

if [ $stage -le 41 ]; then
  datasets=vox1 testsets=vox1
  model=ThinResNet resnet_size=8
#  resnet_size=50
  encoder_type=SAP2 embedding_size=256
  alpha=0
  block_type=basic
  input_norm=Mean
  loss=subarc
  num_center=4
  feat_type=klsp
  sname=dev
  downsample=k1
  batch_size=256

  mask_layer=rvec mask_len=5,10
  weight=rclean_max
  scheduler=rop optimizer=sgd
  fast=none1
  expansion=4
  chn=16
  cyclic_epoch=8
  avg_size=5
#  nesterov
  weight_p=0 scale=0.2
  weight_norm=sum

  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
#  for scale in 0.3 0.5 0.8; do
#  for weight_norm in sum ; do
  for lr in 0.1 ; do
  for resnet_size in 10; do
    for seed in 123456 ;do
    for chn in 16 ; do
      if [ $resnet_size -le 34 ];then
        expansion=1
      else
        expansion=4
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
        if [[ $weight_norm != max ]];then
          at_str=${at_str}${weight_norm}
        fi
      elif [ "$mask_layer" = "drop" ];then
        at_str=_${weight}_dp${weight_p}s${scale}
      elif [ "$mask_layer" = "both" ];then
        at_str=_`echo $mask_len | sed  's/,//g'`
      fi
      echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"

      model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_center${num_center}_${chn_str}wde4_vares/${seed}

      python TrainAndTest/train_egs.py \
        --model ${model} --resnet-size ${resnet_size} \
        --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
        --train-test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
        --train-trials trials \
        --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
        --test-dir ${lstm_dir}/data/${testsets}/${feat_type}/test \
        --feat-format kaldi \
        --input-norm ${input_norm} \
        --seed ${seed} --shuffle --random-chunk 200 400 \
        --batch-size ${batch_size} \
        --optimizer ${optimizer} --cyclic-epoch ${cyclic_epoch} --scheduler ${scheduler} \
        --nj 6 --epochs 60 --patience 3 \
        --early-stopping --early-patience 15 --early-delta 0.0001 --early-meta EER \
        --accu-steps 1 \
        --lr ${lr} --base-lr 0.0000005 \
        --milestones 10,20,30,40 \
        --check-path Data/checkpoint/${model_dir} \
        --resume Data/checkpoint/${model_dir}/checkpoint_25.pth \
        --mask-layer ${mask_layer} \
        --mask-len ${mask_len} --init-weight ${weight} --weight-norm ${weight_norm} \
        --weight-p 0 --scale 0.2 \
        --kernel-size 5,5 --stride 2,2 --fast ${fast} \
        --channels ${channels} \
        --block-type ${block_type} --downsample ${downsample} --expansion ${expansion} \
        --input-dim 161 \
        --time-dim 1 --avg-size ${avg_size} \
        --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
        --num-valid 2 \
        --alpha ${alpha} \
        --loss-type ${loss} --s 30 --margin 0.2 --output-subs --num-center ${num_center}\
        --grad-clip 0 \
        --lr-ratio 0.01 --weight-decay 0.0001 \
        --dropout-p 0.1 \
        --gpu-id 1,6 \
        --all-iteraion 0 \
        --extract --cos-sim
    done
    done
    done
#  done
  done
exit
fi

if [ $stage -le 50 ]; then
  datasets=vox2
  model=ThinResNet resnet_size=50
  encoder_type=SAP2 embedding_size=256
  alpha=0
  block_type=basic

  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
  downsample=k1
  batch_size=256

  mask_layer=rvec
  scheduler=rop optimizer=sgd
  fast=none1
  avg_size=5
  expansion=4
  chn=16
  cyclic_epoch=8
#  nesterov
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
  for seed in 123456 123457 123458 ;do
    for resnet_size in 8 ; do
      if [ $resnet_size -le 34 ];then
        expansion=1
      else
        expansion=4
      fi
      if [ $chn -eq 16 ]; then
        channels=16,32,64,128
        chn_str=
      elif [ $chn -eq 32 ]; then
        channels=32,64,128,256
        chn_str=chn32_
      fi

      model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${seed}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_${chn_str}wde5_var

      echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
      python TrainAndTest/train_egs.py \
        --model ${model} --resnet-size ${resnet_size} \
        --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
        --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
        --train-trials trials \
        --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
        --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
        --feat-format kaldi \
        --input-norm ${input_norm} --random-chunk 200 400 \
        --seed ${seed} \
        --optimizer ${optimizer} --scheduler ${scheduler} \
        --cyclic-epoch ${cyclic_epoch} \
        --nj 8--epochs 60 \
        --patience 2 --early-stopping --early-patience 15 \
        --early-delta 0.01 --early-meta EER \
        --accu-steps 1 \
        --lr 0.1 --base-lr 0.000001 \
        --milestones 10,20,30,40 \
        --check-path Data/checkpoint/${model_dir} \
        --resume Data/checkpoint/${model_dir}/checkpoint_39.pth \
        --kernel-size 5,5 --stride 2,2 --fast ${fast} \
        --channels ${channels} \
        --input-dim 161 \
        --block-type ${block_type} --downsample ${downsample} --expansion ${expansion} \
        --batch-size ${batch_size} \
        --time-dim 1 --avg-size ${avg_size} \
        --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
        --num-valid 2 \
        --alpha ${alpha} \
        --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
        --grad-clip 0 \
        --lr-ratio 0.01 --weight-decay 0.00001 \
        --dropout-p 0.1 \
        --gpu-id 4,5 \
        --extract --shuffle --cos-sim
    done
  done
exit
fi

if [ $stage -le 60 ]; then
  datasets=vox2 testset=vox1
  model=ThinResNet resnet_size=50
  encoder_type=SAP2
  alpha=0
  block_type=basic red_ratio=2 expansion=4
  embedding_size=256
  input_norm=Mean batch_size=128 input_dim=161
  loss=arcsoft
  feat_type=klsp
  sname=dev

  mask_layer=rvec
  scheduler=rop optimizer=sgd
  fast=none1
  downsample=k1 chn=16
  avg_size=5
  seed=123456

  for resnet_size in 50 ; do
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

    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_exp${expansion}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_${chn_str}wde5_vares_bashuf2/${seed}

    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --feat-format kaldi --nj 4 --batch-size ${batch_size} --shuffle --batch-shuffle \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --epochs 80 --random-chunk 200 400 \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --early-stopping --early-patience 15 --early-delta 0.01 --early-meta EER \
      --patience 2 --accu-steps 1 \
      --lr 0.1 --base-lr 0.000001 --milestones 10,20,40,50 \
      --kernel-size 5,5 --stride 2 --fast ${fast} \
      --channels ${channels} \
      --block-type ${block_type} --downsample ${downsample} --red-ratio ${red_ratio} --expansion ${expansion} \
      --time-dim 1 --avg-size ${avg_size} \
      --alpha ${alpha} --dropout-p 0.1 \
      --encoder-type ${encoder_type} --embedding-size ${embedding_size} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_10.pth \
      --grad-clip 0 --lr-ratio 0.01 \
      --weight-decay 0.00001 \
      --gpu-id 5 \
      --extract --num-valid 2 \
      --cos-sim
  done
  exit
fi


if [ $stage -le 61 ]; then
  model=ThinResNet
  datasets=vox2 feat_type=klsp
  loss=arcsoft
  encod=SAP2 embedding_size=256
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
   CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=6 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417420 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Spectrogram/ResNets/vox2_resnet.yaml --seed=${seed}

#    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Fbank/ResNets/aidata_resnet_mixup.yaml --seed=${seed}
  done
  exit
fi

if [ $stage -le 62 ]; then
  model=ThinResNet
  datasets=vox1 feat_type=klsp
  loss=arcsoft
  encod=SAP2 embedding_size=256
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
    CUDA_VISIBLE_DEVICES=1,1 OMP_NUM_THREADS=6 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417425 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Spectrogram/ResNets/vox1_resnet18.yaml --seed=${seed}

    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=6 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417425 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Spectrogram/ResNets/vox1_resnet34.yaml --seed=${seed}

#    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Fbank/ResNets/aidata_resnet_mixup.yaml --seed=${seed}
  done
  exit
fi