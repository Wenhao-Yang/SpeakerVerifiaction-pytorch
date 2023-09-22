#!/usr/bin/env bash

stage=0
waited=0
while [ $(ps 106034 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
    CUDA_VISIBLE_DEVICES=0,2 OMP_NUM_THREADS=12 torchrun --nproc_per_node=2 --master_port=41725 --nnodes=1 TrainAndTest/train_egs/train_dist.py --train-config=TrainAndTest/wav/resnet/v2/v2_unet.yaml --seed=${seed}
fi
