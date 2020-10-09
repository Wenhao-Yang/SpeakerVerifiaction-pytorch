#!/usr/bin/env bash

stage=7
# voxceleb1
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
if [ $stage -le 0 ]; then
  for name in dev test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
#      --out-set ${name}_noc \
#      --windowsize 0.025 \
#      --filters 64 \
#      --feat-type fbank

     python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_fb24 \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type mel \
      --filters 24 \
      --feat-type fbank

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_fb40 \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type mel \
      --filters 40 \
      --feat-type fbank
  done
fi

if [ $stage -le 1 ]; then
  for name in dev test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
#      --out-set ${name}_noc \
#      --windowsize 0.025 \
#      --filters 64 \
#      --feat-type fbank

#     python Process_Data/Compute_Feat/make_feat.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect \
#      --out-set ${name}_noc \
#      --windowsize 0.02 \
#      --nfft 320 \
#      --feat-type spectrogram

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_dfb24_soft \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type dnn.vox1.soft \
      --filters 24 \
      --feat-type fbank
  done
fi

#stage=100

if [ $stage -le 2 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_fb80 \
      --feat-type fbank \
      --filter-type mel \
      --nfft 512 \
      --filters 80 \
      --windowsize 0.025
  done
fi

#stage=200.0
if [ $stage -le 2 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40 \
      --out-set ${name}_kaldi \
      --feat-type mfcc \
      --filters 40
  done
fi
#stage=100
if [ $stage -le 2 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
      --out-set ${name}_linear \
      --feat-type fbank \
      --filter-type linear
  done
fi

#stage=4
if [ $stage -le 3 ]; then
  for name in reverb babble noise music ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_${name}_fb64/dev \
      --out-set dev_${name}

  done
fi

# vox1 spectrogram 257
if [ $stage -le 4 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect \
      --out-set ${name}_257 \
      --windowsize 0.025 \
      --nfft 512 \
      --feat-type spectrogram
  done
fi
#stage=100
#vox1 spectrogram 161
if [ $stage -le 5 ]; then
  for name in dev test ; do
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

#stage=50
# sitw
if [ $stage -le 6 ]; then
  for name in dev eval ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --out-set ${name} \
      --feat-type spectrogram
  done
fi

# timit
if [ $stage -eq 7 ]; then
#  for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect \
#      --nj 12 \
#      --out-set ${name}_log \
#      --log-scale \
#      --feat-type spectrogram \
#      --nfft 320 \
#      --windowsize 0.02

#    python Process_Data/Compute_Feat/make_feat.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect \
#      --nj 12 \
#      --out-set ${name}_power \
#      --feat-type spectrogram \
#      --nfft 320 \
#      --windowsize 0.02
#  done
  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/timit/spect/train_log \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 12 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 1 \
      --out-set train_log
#    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/timit/egs/spect/train_log

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/timit/spect/train_log \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 12 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 1 \
      --out-set valid_log

#    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/timit/egs/spect/valid_log
  done
fi


stage=2000
if [ $stage -le 8 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_fb24 \
      --feat-type fbank \
      --filter-type mel \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 24
  done
fi

#stage=100
if [ $stage -le 9 ]; then
  for name in train test ; do
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

#stage=100
# libri
if [ $stage -le 11 ]; then
  for name in dev test ; do
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
if [ $stage -le 12 ]; then
  for name in dev test ; do
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
if [ $stage -le 13 ]; then
  for name in dev test ; do
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

if [ $stage -le 20 ]; then
# dev
  for name in test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/aishell2/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/aishell2/spect \
      --nj 20 \
      --out-set ${name}_noc \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

#stage=1000
if [ $stage -le 30 ]; then
#enroll
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect \
      --out-set ${name}_noc \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

if [ $stage -le 40 ]; then
#enroll
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/aiox1_${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/aiox1_spect \
      --out-set ${name} \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

if [ $stage -le 50 ]; then
#enroll
  for name in dev test ; do
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
if [ $stage -le 60 ]; then
#enroll
  for name in dev test ; do
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

if [ $stage -le 70 ]; then
#enroll
  for s in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect/${s}_3w \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
      --out-set ${s}_3w \
      --log-scale \
      --feat-type spectrogram \
      --feat-format kaldi \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 15
  done
fi


if [ $stage -le 73 ]; then
  for s in dev test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect/${s} \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
      --out-set ${s}_log \
      --log-scale \
      --feat-type spectrogram \
      --feat-format kaldi_cmp \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 18
    done

  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox1/spect/dev_log \
      --out-dir ${lstm_dir}/data/vox1/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 384 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set dev_log
    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/vox1/egs/spect/dev_log
Process_Data/Compute_Feat/sort_scp.sh /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/egs/spect/dev_log
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox1/spect/dev_log \
      --out-dir ${lstm_dir}/data/vox1/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 384 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set valid_log
    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/vox1/egs/spect/valid_log
  done

fi

stage=1000

if [ $stage -le 74 ]; then
  for s in test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/${s} \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb \
      --out-set ${s}_fb64 \
      --feat-type fbank \
      --filters 64 \
      --log-scale \
      --feat-format kaldi \
      --nfft 512 \
      --windowsize 0.025 \
      --nj 12
    done
fi

#stage=100

if [ $stage -le 75 ]; then
  for s in test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect/${s} \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
      --out-set ${s}_power_spk \
      --feat-type spectrogram \
      --feat-format kaldi \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 18
  done
fi

#stage=1000
if [ $stage -le 80 ]; then
  for s in dev test; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/${s} \
      --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox2/spect \
      --out-set ${s}_power \
      --feat-type spectrogram \
      --feat-format kaldi \
      --nfft 320 \
      --windowsize 0.02 \
      --nj 24
    done
fi

if [ $stage -le 101 ]; then
  for s in dev test; do
    python Process_Data/Compute_Feat/conver2lmdb.py \
      --data-dir ${lstm_dir}/data/vox1/spect/${s}_power \
      --out-dir ${lstm_dir}/data/vox1/lmdb/spect \
      --out-set ${s}_power
  done
fi

if [ $stage -le 102 ]; then
  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox1/spect/${s}_power_257 \
      --out-dir ${lstm_dir}/data/vox1/egs/spect \
      --feat-type spectrogram \
      --train \
      --out-set dev_power_257

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox1/spect/${s}_power_257 \
      --out-dir ${lstm_dir}/data/vox1/egs/spect \
      --feat-type spectrogram \
      --out-set valid_power_257
  done
fi

if [ $stage -le 110 ]; then
  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/spect/dev_4w \
      --out-dir ${lstm_dir}/data/cnceleb/egs/spect \
      --feat-type spectrogram \
      --train \
      --domain \
      --input-per-spks 192 \
      --feat-format kaldi \
      --out-set dev_4w

    mv ${lstm_dir}/data/cnceleb/egs/spect/dev_4w/feats.scp ${lstm_dir}/data/cnceleb/egs/spect/dev_4w/feats.scp.back
    sort -k 3 ${lstm_dir}/data/cnceleb/egs/spect/dev_4w/feats.scp.back > ${lstm_dir}/data/cnceleb/egs/spect/dev_4w/feats.scp

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/cnceleb/spect/dev_4w \
      --out-dir ${lstm_dir}/data/cnceleb/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 192 \
      --feat-format kaldi \
      --domain \
      --out-set valid_4w

    mv ${lstm_dir}/data/cnceleb/egs/spect/valid_4w/feats.scp ${lstm_dir}/data/cnceleb/egs/spect/valid_4w/feats.scp.back
    sort -k 3 ${lstm_dir}/data/cnceleb/egs/spect/valid_4w/feats.scp.back > ${lstm_dir}/data/cnceleb/egs/spect/valid_4w/feats.scp
  done
fi

#stage=1000
if [ $stage -le 111 ]; then
#enroll
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/cnceleb/${name} \
      --out-dir ${lstm_dir}/data/cnceleb/spect \
      --out-set ${name} \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02 \
      --feat-format kaldi \
      --nj 18
  done
fi


if [ $stage -le 120 ]; then
  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/timit/spect/train_power \
      --out-dir ${lstm_dir}/data/timit/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 224 \
      --feat-format kaldi \
      --num-valid 1 \
      --out-set train_power_v2

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/timit/egs/spect/train_power_v2

#    mv ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp.back > ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp

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

#stage=1000
if [ $stage -le 130 ]; then
  for s in dev test ; do
    python Process_Data/Compute_Feat/make_feat.py \
        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/${s}_8k_wav \
        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
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

#    python Process_Data/Compute_Feat/make_feat.py \
#        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/8k_radio_v2/${s}_1w \
#        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
#        --out-set ${s}_8k_radio_v2_1w \
#        --feat-type spectrogram \
#        --lowfreq 300 \
#        --highfreq 3000 \
#        --bandpass \
#        --feat-format kaldi \
#        --nfft 160 \
#        --windowsize 0.02 \
#        --log-scale \
#        --nj 18


  done
#  python Process_Data/Compute_Feat/make_feat.py \
#        --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/test_8k_radio_v3 \
#        --out-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect \
#        --out-set test_8k_radio_v3 \
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

if [ $stage -le 131 ]; then
  for s in dev test ; do
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
if [ $stage -le 132 ]; then

  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/army/spect/dev_8k \
      --out-dir ${lstm_dir}/data/army/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 224 \
      --feat-format kaldi \
      --num-valid 4 \
      --out-set dev_v1

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/army/egs/spect/dev_v1

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/army/spect/dev_8k \
      --out-dir ${lstm_dir}/data/army/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 224 \
      --feat-format kaldi \
      --num-valid 4 \
      --out-set valid_v1

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/army/egs/spect/valid_v1
  done
fi

if [ $stage -le 140 ]; then
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

if [ $stage -le 150 ]; then
#enroll
  for name in dev ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir ${lstm_dir}/data/vox2/${name} \
      --out-dir ${lstm_dir}/data/vox2/spect \
      --out-set ${name}_power \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02 \
      --feat-format kaldi \
      --nj 18
  done
fi


if [ $stage -le 151 ]; then
  for s in dev ; do
    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox2/spect/dev_power \
      --out-dir ${lstm_dir}/data/vox2/egs/spect \
      --feat-type spectrogram \
      --train \
      --input-per-spks 192 \
      --feat-format kaldi \
      --num-valid 2 \
      --out-set dev_power

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/vox2/egs/spect/dev_power

#    mv ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp.back > ${lstm_dir}/data/timit/egs/spect/train_power/feats.scp

    python Process_Data/Compute_Feat/make_egs.py \
      --data-dir ${lstm_dir}/data/vox2/spect/dev_power \
      --out-dir ${lstm_dir}/data/vox2/egs/spect \
      --feat-type spectrogram \
      --input-per-spks 192 \
      --feat-format kaldi \
      --num-valid 2 \
      --out-set valid_power

    Process_Data/Compute_Feat/sort_scp.sh ${lstm_dir}/data/vox2/egs/spect/valid_power
#    mv ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp.back
#    sort -k 2 ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp.back > ${lstm_dir}/data/timit/egs/spect/valid_power/feats.scp
  done
fi

if [ $stage -le 200 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_soft \
      --feat-type fbank \
      --filter-type dnn.timit.soft \
      --log-scale \
      --nfft 320 \
      --windowsize 0.02 \
       --filters 23
  done

    for name in train test ; do
    python Process_Data/Compute_Feat/make_feat.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_arcsoft \
      --feat-type fbank \
      --filter-type dnn.timit.arcsoft \
      --log-scale \
      --nfft 320 \
      --windowsize 0.02 \
       --filters 23
  done
fi
