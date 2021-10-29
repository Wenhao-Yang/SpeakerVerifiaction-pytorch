#!/usr/bin/env bash

stage=300
# voxceleb1
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

# echo $(awk '{n += $2} END{print n}' <utt2num_frames)

waited=0
while [ $(ps 12841 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

# ==================================================   vox 1 & 2   ==================================================

if [ $stage -le 0 ]; then
  dataset=vox1
  #  feat=fb40
  feat=spect
  if [ "$feat" = "pyfb" ]; then
    feat_type=fbank
  elif [ "$feat" = "spect" ]; then
    feat_type=spectrogram
  elif [ "$feat" = "klfb" ]; then
    feat_type=klfb
  fi
  #        --filters ${filters} \
  #      --log-scale \

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} for ${dataset}\033[0m\n"
  for filters in 40; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${dataset}/klsp/dev_aug \
      --out-dir ${lstm_dir}/data/${dataset}/spect \
      --out-set dev_aug_log \
      --log-scale \
      --feat-type ${feat_type} \
      --feat-format kaldi_cmp \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 16
  done
  exit
fi
#exit
#stage=1000
if [ $stage -le 1 ]; then
  dataset=vox2
  #  feat_type=pyfb
  #  dataset=vox1
  feat=klsp
  feat_type=klsp

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} egs for ${dataset}\033[0m\n"
  #  for s in dev_log dev_aug_1m_log ; do
  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --train \
      --input-per-spks 896 \
      --num-frames 600 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set ${s}

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --num-frames 600 \
      --input-per-spks 896 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set ${s}_valid
  done
  exit
fi
#exit

#stage=200.0
if [ $stage -le 2 ]; then
  dataset=vox1
  feat=klfb

  if [ "$feat" = "pyfb" ]; then
    feat_type=fbank
  elif [ "$feat" = "spect" ]; then
    feat_type=spectrogram
  elif [ "$feat" = "klfb" ]; then
    feat_type=klfb
  fi

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} egs with kaldi fbank for ${dataset}\033[0m\n"
  for s in combined; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/dev_${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --train \
      --input-per-spks 0 \
      --num-frames 400 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --remove-vad \
      --out-set dev_${s}_fb40_v3

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/dev_${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --num-frames 400 \
      --input-per-spks 0 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --remove-vad \
      --out-set valid_${s}_fb40_v3
  done
  exit
fi

if [ $stage -le 3 ]; then
  for filters in 40; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/vox2/dev \
      --out-dir ${lstm_dir}/data/vox2/mfcc \
      --out-set dev_mf${filters} \
      --filter-type mel \
      --feat-type mfcc \
      --filters ${filters} \
      --numcep ${filters} \
      --feat-format kaldi_cmp \
      --nfft 512 \
      --windowsize 0.025 \
      --stride 0.01 \
      --nj 16
  done

  #  for s in fb40 fb80 ; do
  for s in mf40; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox2/mfcc/dev_${s} \
      --out-dir ${lstm_dir}/data/vox2/egs/mfcc \
      --nj 16 \
      --feat-type fbank \
      --train \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set dev_${s}

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox2/mfcc/dev_${s} \
      --out-dir ${lstm_dir}/data/vox2/egs/mfcc \
      --nj 16 \
      --feat-type fbank \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set valid_${s}

  done

fi
#exit

#stage=100
# vox1 spectrogram 257
if [ $stage -le 4 ]; then
  for filters in 40; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/vox1/dev \
      --out-dir ${lstm_dir}/data/vox1/pyfb \
      --out-set dev_fb${filters} \
      --filter-type mel \
      --feat-type fbank \
      --filters ${filters} \
      --log-scale \
      --feat-format kaldi_cmp \
      --nfft 512 \
      --windowsize 0.025 \
      --nj 16
  done
fi
#exit

#stage=100
#vox1 spectrogram 161
if [ $stage -le 5 ]; then
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --nj 16 \
      --data-dir ${lstm_dir}/data/vox1/${name} \
      --out-dir ${lstm_dir}/data/vox1/spect \
      --out-set ${name} \
      --nfft 320 \
      --windowsize 0.02 \
      --feat-format npy \
      --feat-type spectrogram
  done
fi

#stage=40
# ==================================================   sitw   =====================================================
if [ $stage -le 10 ]; then
  dataset=sitw
  filters=40
  feat=fb40
  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} for ${dataset}\033[0m\n"
  for s in eval; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${dataset}/${s} \
      --out-dir ${lstm_dir}/data/${dataset}/pyfb \
      --out-set ${s}_fb${filters}_ws25 \
      --filter-type mel \
      --feat-type fbank \
      --filters ${filters} \
      --log-scale \
      --feat-format kaldi_cmp \
      --nfft 512 \
      --windowsize 0.025 \
      --nj 12
  done
  exit
fi

# ==================================================   timit   ==================================================
if [ $stage -eq 20 ]; then
  for name in train test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect \
      --nj 12 \
      --out-set ${name}_log \
      --log-scale \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02

    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect \
      --nj 12 \
      --out-set ${name}_power \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

if [ $stage -le 21 ]; then

  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --nj 12 \
      --data-dir ${lstm_dir}/data/timit/spect/train_log \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 192 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 1 \
      --out-set train_log

    python Process_Data/Compute_Feat/make_egs.py \
      --nj 12 \
      --data-dir ${lstm_dir}/data/timit/spect/train_log \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 192 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 1 \
      --out-set valid_log

  done
fi

#stage=1000
if [ $stage -le 22 ]; then
  for name in train test; do
    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
    #      --out-set ${name}_fb40_20 \
    #      --filter-type mel \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 40

    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
    #      --out-set ${name}_fb40_dnn_20 \
    #      --filter-type dnn.timit \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 40
    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
    #      --out-set ${name}_fb30 \
    #      --filter-type mel \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 30
    #
    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
    #      --out-set ${name}_dfb30_fix \
    #      --filter-type dnn.timit.fix \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 30

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_dfb24_var_f1 \
      --filter-type dnn.timit.var \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 24
  done
fi

#stage=100
#if [ $stage -le 10 ]; then
#  for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_mfcc_20 \
#      --filter-type mel \
#      --feat-type mfcc \
#      --nfft 320 \
#      --lowfreq 20 \
#      --windowsize 0.02 \
#      --filters 30 \
#      --numcep 24
#
#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_mfcc_dnn_20 \
#      --filter-type dnn.timit \
#      --feat-type mfcc \
#      --nfft 320 \
#      --lowfreq 20 \
#      --windowsize 0.02 \
#      --filters 30 \
#      --numcep 24
#  done
#fi

if [ $stage -le 23 ]; then
  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/timit/spect/train_power \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 224 \
      --feat-format kaldi \
      --num-valid 1 \
      --out-set train_power_v2

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/timit/spect/train_power \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 224 \
      --feat-format kaldi \
      --num-valid 1 \
      --out-set valid_power_v2

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/timit/egs/spect/valid_power_v2
    #    mv ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp.back
    #    sort -k 2 ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp.back > ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp
  done
fi

#stage=100
# ==================================================   librispeech   ==================================================

# libri
if [ $stage -le 25 ]; then
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect \
      --out-set ${name}_noc \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

#stage=100
if [ $stage -le 26 ]; then
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb \
      --out-set ${name}_fb80 \
      --filter-type mel \
      --feat-type fbank \
      --nfft 512 \
      --windowsize 0.025 \
      --filters 80

    #     python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri \
    #      --out-set ${name}_fb40_dnn_20 \
    #      --filter-type dnn.timit \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 40
  done
fi

#stage=100
if [ $stage -le 27 ]; then
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb \
      --out-set ${name}_lfb24 \
      --filter-type linear \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 24

    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
    #      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb \
    #      --out-set ${name}_dfb24_var \
    #      --filter-type dnn.libri.var \
    #      --feat-type fbank \
    #      --nfft 320 \
    #      --windowsize 0.02 \
    #      --filters 24
  done
fi

# ==================================================   aishell2   ==================================================

if [ $stage -le 30 ]; then
  # dev
  dataset=aishell2
  #  dataset=magic
  feat=pyfb
  filters=40
  feat_name=fb${filters}

  if [ "$feat" = "pyfb" ]; then
    feat_type=fbank
  elif [ "$feat" = "spect" ]; then
    feat_type=spectrogram
  fi

  for name in test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${dataset}/${name} \
      --out-dir ${lstm_dir}/data/${dataset}/${feat} \
      --out-set ${name}_fb${filters}_ws25 \
      --filter-type mel \
      --nj 12 \
      --feat-type ${feat_type} \
      --filters ${filters} \
      --log-scale \
      --feat-format kaldi_cmp \
      --nfft 512 \
      --windowsize 0.025
  done
  exit
fi

if [ $stage -le 31 ]; then
  # dev
  dataset=aishell2
  #  dataset=aidata
  python Process_Data/Compute_Feat/make_egs.py \
    --data-dir ${lstm_dir}/data/${dataset}/spect/dev_log \
    --out-dir ${lstm_dir}/data/${dataset}/egs/spect \
    --feat-type spectrogram \
    --train \
    --input-per-spks 768 \
    --feat-format kaldi \
    --out-format kaldi_cmp \
    --num-valid 2 \
    --out-set dev_log

  python Process_Data/Compute_Feat/make_egs.py \
    --data-dir ${lstm_dir}/data/${dataset}/spect/dev_log \
    --out-dir ${lstm_dir}/data/${dataset}/egs/spect \
    --feat-type spectrogram \
    --input-per-spks 768 \
    --feat-format kaldi \
    --out-format kaldi_cmp \
    --num-valid 2 \
    --out-set valid_log
  exit
fi

#stage=1000
# ==================================================   aishell2   ==================================================

if [ $stage -le 40 ]; then
  #enroll
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/cnceleb/${name} \
      --out-dir ${lstm_dir}/data/cnceleb/spect \
      --out-set ${name} \
      --feat-type spectrogram \
      --feat-format npy \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 20
  done
fi

#stage=100
# ==================================================   army   ==================================================
if [ $stage -le 50 ]; then
  #enroll
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/${name} \
      --out-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/spect \
      --out-set ${name} \
      --feat-type spectrogram \
      --feat-format npy \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 20
  done
fi

#stage=1000
if [ $stage -le 51 ]; then
  for s in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/${s}_8k_wav \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
      --out-set ${s}_8k_log \
      --feat-type spectrogram \
      --feat-format kaldi \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 16
  done
  exit
  #
  #    python Process_Data/Compute_Feat/make_feat.py \
  #      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/8k_radio_v3/${s} \
  #      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
  #      --out-set ${s}_8k_radio_v3_log \
  #      --feat-type spectrogram \
  #      --lowfreq 300 \
  #      --highfreq 3000 \
  #      --bandpass \
  #      --feat-format kaldi \
  #      --nfft 160 \
  #      --windowsize 0.02 \
  #      --log-scale \
  #      --nj 18
  #
  #  done

  for s in dev test; do
    #    python Process_Data/Compute_Feat/make_feat.py \
    #      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k/${s}_8k \
    #      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
    #      --out-set ${s}_8k_log \
    #      --feat-type spectrogram \
    #      --feat-format kaldi \
    #      --nfft 160 \
    #      --windowsize 0.02 \
    #      --log-scale \
    #      --nj 18

    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k_radio_v3/${s}_8k-radio-v3 \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
      --out-set ${s}_8k_radio_v3_log \
      --feat-type spectrogram \
      --lowfreq 300 \
      --highfreq 3000 \
      --bandpass \
      --feat-format kaldi \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 18

  done
  exit

#  for s in dev ; do
#    python Process_Data/Compute_Feat/make_egs.py \
#      --data-dir ${lstm_dir}/data/army/spect/vox1_dev_clear_radio \
#      --out-dir ${lstm_dir}/data/army/egs/spect \
#      --feat-type spectrogram \
#      --train \
#      --input-per-spks 384 \
#      --feat-format kaldi \
#      --num-valid 4 \
#      --out-set vox1_dev_clear_radio
#
#    mv ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp.back > ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp
#
#    python Process_Data/Compute_Feat/make_egs.py \
#      --data-dir ${lstm_dir}/data/army/spect/vox1_dev_clear_radio \
#      --out-dir ${lstm_dir}/data/army/egs/spect \
#      --feat-type spectrogram \
#      --input-per-spks 384 \
#      --feat-format kaldi \
#      --num-valid 4 \
#      --out-set vox1_valid_clear_radio
#
#    mv ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp.back > ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp
#  done
fi

if [ $stage -le 52 ]; then
  for s in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k/${s}_8k \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
      --out-set ${s}_8k \
      --feat-type spectrogram \
      --lowfreq 300 \
      --highfreq 3000 \
      --bandpass \
      --feat-format kaldi \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 18

    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k_radio_v3/${s}_8k-radio-v3 \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
      --out-set ${s}_8k_radio_v3 \
      --feat-type spectrogram \
      --lowfreq 300 \
      --highfreq 3000 \
      --bandpass \
      --feat-format kaldi \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 18
  done

#  python Process_Data/Compute_Feat/make_feat.py \
#        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k_musan/dev_musan_dev \
#        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
#        --out-set dev_8k_musan \
#        --feat-type spectrogram \
#        --lowfreq 300 \
#        --highfreq 3000 \
#        --bandpass \
#        --feat-format kaldi \
#        --nfft 160 \
#        --windowsize 0.02 \
#        --log-scale \
#        --nj 18

#  for s in dev test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/${s}_8k_wav \
#        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
#        --out-set ${s}_8k \
#        --feat-type spectrogram \
#        --lowfreq 300 \
#        --highfreq 3000 \
#        --bandpass \
#        --feat-format kaldi \
#        --nfft 160 \
#        --windowsize 0.02 \
#        --log-scale \
#        --nj 18
#  done

#  for s in dev ; do
#    python Process_Data/Compute_Feat/make_egs.py \
#      --data-dir ${lstm_dir}/data/army/spect/vox1_dev_clear_radio \
#      --out-dir ${lstm_dir}/data/army/egs/spect \
#      --feat-type spectrogram \
#      --train \
#      --input-per-spks 384 \
#      --feat-format kaldi \
#      --num-valid 4 \
#      --out-set vox1_dev_clear_radio
#
#    mv ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp.back > ${lstm_dir}/data/army/egs/spect/vox1_dev_clear_radio/feats.scp
#
#    python Process_Data/Compute_Feat/make_egs.py \
#      --data-dir ${lstm_dir}/data/army/spect/vox1_dev_clear_radio \
#      --out-dir ${lstm_dir}/data/army/egs/spect \
#      --feat-type spectrogram \
#      --input-per-spks 384 \
#      --feat-format kaldi \
#      --num-valid 4 \
#      --out-set vox1_valid_clear_radio
#
#    mv ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp.back > ${lstm_dir}/data/army/egs/spect/vox1_valid_clear_radio/feats.scp
#  done
fi

#stage=2100
if [ $stage -le 53 ]; then

  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/army/spect/dev_8k \
      --out-dir ${lstm_dir}/data/army/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 4 \
      --out-set dev_v2

    #    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/army/egs/spect/dev_v2

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/army/spect/dev_8k \
      --out-dir ${lstm_dir}/data/army/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 512 \
      --out-format kaldi_cmp \
      --feat-format kaldi \
      --num-valid 4 \
      --out-set valid_v2

    #    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/army/egs/spect/valid_v2
  done
fi

if [ $stage -le 54 ]; then
  for s in dev; do
    #    python Process_Data/Compute_Feat/make_feat.py \
    #        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/${s}_8k_v4 \
    #        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
    #        --out-set ${s}_8k_v4 \
    #        --feat-type spectrogram \
    #        --feat-format kaldi_cmp \
    #        --nfft 160 \
    #        --windowsize 0.02 \
    #        --log-scale \
    #        --nj 6
    #
    #    python Process_Data/Compute_Feat/make_feat.py \
    #        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/8k/${s}_8k_v4 \
    #        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/aishell2/spect \
    #        --out-set ${s}_8k_v4 \
    #        --feat-type spectrogram \
    #        --feat-format kaldi_cmp \
    #        --nfft 160 \
    #        --windowsize 0.02 \
    #        --log-scale \
    #        --nj 6

    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/dev_8k_7h_v4 \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/spect \
      --out-set dev_8k_7h_v4 \
      --feat-type spectrogram \
      --feat-format kaldi_cmp \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 18

    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/dev_8k_7h_v3 \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/spect \
      --out-set dev_8k_7h_v3 \
      --feat-type spectrogram \
      --feat-format kaldi_cmp \
      --lowfreq 300 \
      --highfreq 3000 \
      --bandpass \
      --nfft 160 \
      --windowsize 0.02 \
      --log-scale \
      --nj 18

  done
fi

#stage=10000
if [ $stage -le 55 ]; then
  python Process_Data/Compute_Feat/make_feat.py \
    --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/radio/example_8k \
    --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/radio/spect \
    --out-set example_8k \
    --feat-type spectrogram \
    --lowfreq 300 \
    --highfreq 3000 \
    --bandpass \
    --feat-format kaldi \
    --nfft 160 \
    --windowsize 0.02 \
    --log-scale \
    --nj 4
fi

# =========================================   cnceleb   =========================================
if [ $stage -le 60 ]; then
  #enroll
  datasets=cnceleb
  echo -e "\n\033[1;4;31m Stage ${stage}: Making log spectrogram for ${datasets}\033[0m\n"
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${datasets}/${name} \
      --out-dir ${lstm_dir}/data/${datasets}/spect \
      --out-set ${name}_log \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02 \
      --feat-format kaldi_cmp \
      --log-scale \
      --nj 10
  done
  exit
fi

if [ $stage -le 62 ]; then
  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/spect/dev_4w \
      --out-dir ${lstm_dir}/data/cnceleb/egs/spect \
      --feat-type spectrogram \
      --train \
      --domain \
      --input-per-spks 192 \
      --feat-format kaldi \
      --out-set dev_4w

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/spect/dev_4w \
      --out-dir ${lstm_dir}/data/cnceleb/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 192 \
      --feat-format kaldi \
      --domain \
      --out-set valid_4w

  done
fi

if [ $stage -le 63 ]; then
  #enroll
  datasets=cnceleb
  echo -e "\n\033[1;4;31m Stage ${stage}: Making log fbank for ${datasets}\033[0m\n"
  num_filters=40
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${datasets}/${name} \
      --out-dir ${lstm_dir}/data/${datasets}/pyfb \
      --out-set ${name}_fb${num_filters}_ws25 \
      --filter-type mel \
      --feat-type fbank \
      --filters ${num_filters} \
      --log-scale \
      --feat-format kaldi_cmp \
      --nfft 512 \
      --windowsize 0.025 \
      --nj 12
  done
  exit
fi

if [ $stage -le 64 ]; then
  datasets=cnceleb
  echo -e "\n\033[1;4;31m Stage ${stage}: Making log fbank egs for ${datasets}\033[0m\n"
  for s in dev; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/pyfb/dev_fb40_ws25 \
      --out-dir ${lstm_dir}/data/cnceleb/egs/pyfb \
      --feat-type fbank \
      --train \
      --domain \
      --input-per-spks 512 \
      --num-frames 400 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --remove-vad \
      --out-set dev_fb40_ws25

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/pyfb/dev_fb40_ws25 \
      --out-dir ${lstm_dir}/data/cnceleb/egs/pyfb \
      --feat-type fbank \
      --input-per-spks 512 \
      --feat-format kaldi \
      --num-frames 400 \
      --domain \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --remove-vad \
      --out-set valid_fb40_ws25
  done
  exit
fi

#stage=1000
#stage=10000
if [ $stage -le 200 ]; then
  for name in train test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_soft \
      --feat-type fbank \
      --feat-format kaldi_cmp \
      --filter-type dnn.timit.soft \
      --log-scale \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 23
  done

#    for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
#      --out-set ${name}_arcsoft \
#      --feat-type fbank \
#      --filter-type dnn.timit.arcsoft \
#      --log-scale \
#      --nfft 320 \
#      --windowsize 0.02 \
#       --filters 23
#  done
fi

#stage=10000
if [ $stage -le 210 ]; then
  for name in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/pyfb \
      --out-set ${name}_fb24_soft \
      --feat-type fbank \
      --feat-format kaldi_cmp \
      --filter-type dnn.vox1.soft \
      --log-scale \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 23
  done

#    for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
#      --out-set ${name}_arcsoft \
#      --feat-type fbank \
#      --filter-type dnn.timit.arcsoft \
#      --log-scale \
#      --nfft 320 \
#      --windowsize 0.02 \
#       --filters 23
#  done
fi

if [ $stage -le 230 ]; then
  datasets=army
  for s in aishell2 vox1; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${datasets}/spect/${s}_dev_8k \
      --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 324 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 4 \
      --out-set ${s}_dev_8k_v2

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${datasets}/spect/${s}_dev_8k \
      --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 4 \
      --out-set ${s}_valid_8k_v2

  done
fi

if [ $stage -le 231 ]; then
  datasets=army
  for s in vox; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${datasets}/spect/${s}_dev_8k_v4 \
      --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 4 \
      --out-set ${s}_dev_8k_v4

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${datasets}/spect/${s}_dev_8k_v4 \
      --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 512 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 4 \
      --out-set ${s}_valid_8k_v4

  done
fi

if [ $stage -le 232 ]; then
  datasets=army

  python Process_Data/Compute_Feat/make_egs.py \
    --nj 12 \
    --data-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v5_log \
    --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
    --feat-type spectrogram \
    --train \
    --input-per-spks 768 \
    --feat-format kaldi \
    --out-format kaldi_cmp \
    --num-valid 2 \
    --out-set dev_8k_v5_log

  python Process_Data/Compute_Feat/make_egs.py \
    --nj 12 \
    --data-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v5_log \
    --out-dir ${lstm_dir}/data/${datasets}/egs/spect \
    --feat-type spectrogram \
    --input-per-spks 768 \
    --feat-format kaldi \
    --out-format kaldi_cmp \
    --num-valid 2 \
    --out-set valid_8k_v5_log
  exit

fi

if [ $stage -le 250 ]; then
  dataset=aidata
  for name in dev_8k; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/${dataset}/${name} \
      --out-dir ${lstm_dir}/data/${dataset}/spect \
      --out-set ${name}_log \
      --feat-type spectrogram \
      --feat-format kaldi_cmp \
      --log-scale \
      --nfft 160 \
      --windowsize 0.02
  done
  exit
fi


if [ $stage -le 300 ]; then
  dataset=vox1
  #  feat_type=pyfb
  #  dataset=vox1
  feat=klfb
  feat_type=klfb

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} egs for ${dataset}\033[0m\n"
  #  for s in dev_log dev_aug_1m_log ; do
  for s in dev_auged1_fb40; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --train \
      --input-per-spks 128 \
      --num-frames 300 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --enhance \
      --sets none reverb music noise babble \
      --out-set ${s}_pair

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/${dataset}/${feat}/${s} \
      --out-dir ${lstm_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --num-frames 300 \
      --input-per-spks 128 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --enhance \
      --sets none reverb music noise babble \
      --out-set ${s}_pair_valid
  done
  exit
fi