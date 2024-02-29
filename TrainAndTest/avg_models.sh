#!/usr/bin/env bash

stage=0
waited=0
while [ `ps 99278 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10

lstm_dir=/home/yangwenhao/project/lstm_speaker_verification


if [ $stage -le 0 ]; then
    common_path=ECAPA_brain/Mean_batch96_SASP2_em192_official_2s/arcsoft_adam_cyclic/vox1
    for model_name in ecapa_aug53 ecapa_aug53_dp111 ecapa_aug53_attenoise10100 ecapa_aug53_attenoise10100_prob08 ecapa_aug53_pattenoise10100_prob08 ecapa_aug53_noise10100_prob08 ecapa_aug53_burr10 ecapa_aug53_inspecaug05 ecapa_aug53_dp111_attenoise10100 ecapa_aug53_radionoise1010 ecapa_aug53_radionoise10100; do
    echo -e "\n\033[1;4;31m Stage${stage}: Average model: ${model_name} \033[0m\n"
        for seed in 1234 1235 1236 ; do
            if [[ $model_name == ecapa_aug53 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53/${seed}
            elif [[ $model_name == ecapa_aug53_dp111 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_dp111/${seed}
            elif [[ $model_name == ecapa_aug53_burr10 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_burr1210/${seed}
            elif [[ $model_name == ecapa_aug53_magdp111 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_mag111/${seed}
            elif [[ $model_name == ecapa_aug53_dp111_before ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_dp111_before/${seed}
            elif [[ $model_name == ecapa_aug53_multilayer10 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_attenoise_multilayers1/${seed}
            elif [[ $model_name == ecapa_aug53_inspecaug05 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_inspecaug05/${seed}
            elif [[ $model_name == ecapa_aug53_noise10100_prob08 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_noise10100_prob08/${seed}
            elif [[ $model_name == ecapa_aug53_attenoise10100_prob08 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_attenoise10100_prob08/${seed}
            elif [[ $model_name == ecapa_aug53_pattenoise10100_prob08 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_pattenoise10100_prob08/${seed}
            elif [[ $model_name == ecapa_aug53_dp111_attenoise10100 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_dp111_attenoise10100/${seed}
            elif [[ $model_name == ecapa_aug53_attenoise10100 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_attenoise10100/${seed}
            elif [[ $model_name == ecapa_aug53_radionoise1010 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_radionoise/${seed}
            elif [[ $model_name == ecapa_aug53_radionoise10100 ]];then
                model_dir=${common_path}/wave_fb80_dist_aug53_radionoise10100/${seed}
            fi
            
            python -W ignore TrainAndTest/train_egs/average_model.py \
                --check-path Data/checkpoint/${model_dir}
        done
    done
 exit
fi