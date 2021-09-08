#!/usr/bin/env bash

stage=0
waited=0
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
while [ $(ps 15414 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  model=TDNN_v5
  dataset=cnceleb
  train_set=cnceleb
  feat_type=pyfb
  feat=fb40_ws25
  loss=arcsoft
  encoder_type=STAP
  embedding_size=256
  block_type=basic
  echo -e "\n\033[1;4;31m stage${stage} Training ${model}_${encoder_type} in ${train_set}_${test_set} with ${loss}\033[0m\n"

  python Lime/output_extract.py \
    --model ${model} \
    --start-epochs 50 \
    --epochs 50 \
    --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
    --train-set-name ${train_set} \
    --input-norm Mean \
    --stride 1 \
    --channels 512,512,512,512,1500 \
    --encoder-type ${encoder_type} \
    --block-type ${block_type} \
    --embedding-size ${embedding_size} \
    --alpha 0 \
    --loss-type ${loss} \
    --dropout-p 0.0 \
    --check-path Data/checkpoint/TDNN_v5/cnceleb/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em256_wde3_var/checkpoint_60.pth \
    --extract-path Data/logits/TDNN_v5/cnceleb/pyfb_egs_baseline/arcsoft/featfb40_ws25_inputMean_STAP_em256_wde3_var/epoch_60_var \
    --gpu-id 0 \
    --margin 0.15 \
    --s 30 \
    --sample-utt 12000
  exit
fi
