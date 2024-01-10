#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: dataset.py
@Time: 2023/02/23 18:22
@Overview: 
'''
import json
import os
import torch
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, ScriptTrainDataset, ScriptValidDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset, GenderDataset, LmdbTrainDataset, LmdbValidDataset, Hdf5TrainDataset, Hdf5ValidDataset

from Process_Data.audio_processing import BandPass, ConcateNumInput, MelFbank, totensor, PadCollate3d, stretch
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate, read_WaveInt, read_WaveFloat
import torchvision.transforms as transforms
from kaldiio import load_mat
import numpy as np


def SubDatasets(config_args):
    transform = transforms.Compose([
        totensor()
    ])

    if config_args['test_input'] == 'var':
        transform_V = transforms.Compose([
            ConcateOrgInput(
                remove_vad=config_args['remove_vad'], feat_type=config_args['feat_format']),
        ])
    elif config_args['test_input'] == 'fix':
        transform_V = transforms.Compose([
            ConcateVarInput(remove_vad=config_args['remove_vad'], num_frames=config_args['chunk_size'],
                            frame_shift=config_args['frame_shift'],
                            feat_type=config_args['feat_format']),
        ])

    if config_args['log_scale']:
        transform.transforms.append(tolog())
        transform_V.transforms.append(tolog())

    loader_types = {'kaldi': load_mat, 'wav': load_mat, 'npy': np.load}
    file_loader = loader_types[config_args['feat_format']]

    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    if torch.distributed.is_initialized() and torch.distributed.get_rank() > 0:
        verbose = 0
    else:
        verbose = 1
    train_dir = EgsDataset(dir=config_args['train_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                           transform=transform, batch_size=config_args['batch_size'],
                           random_chunk=config_args['random_chunk'],
                           verbose=verbose,
                           domain=return_domain)

    valid_dir = EgsDataset(dir=config_args['valid_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                           transform=transform,
                           verbose=verbose
                           )

    feat_type = 'kaldi'
    if config_args['feat_format'] == 'wav':
        file_loader = read_WaveInt if config_args['feat'] == 'int' else read_WaveFloat
        feat_type = 'wav'

    train_extract_dir = KaldiExtractDataset(dir=config_args['train_test_dir'],
                                            transform=transform_V,
                                            filer_loader=file_loader, feat_type=feat_type,
                                            trials_file=config_args['train_trials'])

    return train_dir, valid_dir, train_extract_dir


def SubScriptDatasets(config_args):

    if config_args['test_input'] == 'var':
        transform_V = transforms.Compose([
            ConcateOrgInput(
                remove_vad=config_args['remove_vad'])  # , feat_type=config_args['feat_format']),
        ])
    elif config_args['test_input'] == 'fix':
        transform_V = transforms.Compose([
            ConcateVarInput(remove_vad=config_args['remove_vad'], num_frames=config_args['chunk_size'],
                            frame_shift=config_args['chunk_size'],
                            feat_type=config_args['feat_format']),
        ])

    # if config_args['log_scale']:
    #     transform.transforms.append(tolog())
    #     transform_V.transforms.append(tolog())
    # loader_types = {'kaldi': load_mat, 'wav': load_mat, 'npy': np.load}
    # file_loader = loader_types[config_args['feat_format']]

    feat_type = 'kaldi'
    if config_args['feat_format'] == 'npy':
        file_loader = np.load
    elif config_args['feat_format'] in ['kaldi', 'klfb', 'klsp']:
        # file_loader = kaldi_io.read_mat
        file_loader = load_mat
    elif config_args['feat_format'] == 'wav':
        file_loader = read_WaveInt if config_args['wav_type'] == 'int' else read_WaveFloat
        feat_type = 'wav'

    remove_vad = False if 'remove_vad' not in config_args else config_args['remove_vad']
    transform = transforms.Compose([
        ConcateNumInput(num_frames=config_args['chunk_size'], remove_vad=remove_vad,
                        feat_type=feat_type),
        totensor()
    ])

    if 'bandpass' in config_args:
        band_pass_prob = config_args['band_pass_prob'] if 'band_pass_prob' in config_args else 0.2
        order = 15 if 'band_order' not in config_args else config_args['band_order']

        transform.transforms.insert(0, BandPass(low=config_args['bandpass'][0],
                                                high=config_args['bandpass'][1:],
                                                sr=config_args['sr'], order=order,
                                                band_pass_prob=band_pass_prob))

    if 'trans_fbank' in config_args and config_args['trans_fbank']:
        transform.transforms.append(
            MelFbank(num_filter=config_args['input_dim']))
        transform_V.transforms.append(
            MelFbank(num_filter=config_args['input_dim']))

    domain = config_args['domain'] if 'domain' in config_args else False
    sample_type = 'half_balance' if 'sample_type' not in config_args else config_args['sample_type']

    sample_score = config_args['sample_score'] if 'sample_score' in config_args else ''

    vad_select = False if 'vad_select' not in config_args else config_args['vad_select']
    verbose = 1 if torch.distributed.is_initialized(
    ) and torch.distributed.get_rank() == 0 else 0
    segment_shift = config_args['segment_shift'] if 'segment_shift' in config_args else config_args['num_frames']
    min_frames = 50 if 'min_frames' not in config_args else config_args['min_frames']
    if config_args['feat_format'] == 'wav' and 'trans_fbank' not in config_args:
        min_frames *= config_args['sr'] / 100
        min_frames = int(min_frames)

    if 'feat_type' in config_args and config_args['feat_type'] == 'lmdb':
        print('Create Lmdb Dataset...')
        train_dir = LmdbTrainDataset(dir=config_args['train_dir'], samples_per_speaker=config_args['input_per_spks'],
                                     transform=transform, num_valid=config_args['num_valid'],
                                     feat_type='wav', sample_type=sample_type,
                                     segment_len=config_args['num_frames'],
                                     segment_shift=segment_shift,
                                     min_frames=min_frames, verbose=verbose,
                                     return_uid=False)

        valid_dir = LmdbValidDataset(train_dir.valid_set, spk_to_idx=train_dir.spk_to_idx,
                                    reader=train_dir.reader, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                                    verbose=verbose,
                                    transform=transform)

    elif 'feat_type' in config_args and config_args['feat_type'] == 'hdf5':
        print('Create HDF5 Dataset...')
        train_dir = Hdf5TrainDataset(dir=config_args['train_dir'], samples_per_speaker=config_args['input_per_spks'], 
                                     transform=transform, num_valid=config_args['num_valid'],
                                     feat_type='wav', sample_type=sample_type,
                                     segment_len=config_args['num_frames'], 
                                     segment_shift=segment_shift,
                                     min_frames=min_frames, verbose=verbose,
                                     return_uid=False)
        
        valid_dir = Hdf5ValidDataset(train_dir.valid_set, spk_to_idx=train_dir.spk_to_idx,
                                    hdf5_file=train_dir.hdf5_file, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                                    verbose=verbose,
                                    transform=transform)
    else:
        save_dir = config_args['check_path'] if 'save_data_dir' not in config_args else config_args['save_data_dir']
        train_dir = ScriptTrainDataset(dir=config_args['train_dir'], samples_per_speaker=config_args['input_per_spks'], loader=file_loader,
                                    transform=transform, num_valid=config_args['num_valid'], domain=domain,
                                    vad_select=vad_select, sample_type=sample_type,
                                    feat_type=feat_type, verbose=verbose,
                                    save_dir=save_dir,
                                    sample_score=sample_score,
                                    segment_len=config_args['num_frames'],
                                    segment_shift=segment_shift,
                                    min_frames=min_frames)

        valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader, spk_to_idx=train_dir.spk_to_idx,
                                    dom_to_idx=train_dir.dom_to_idx, valid_utt2dom_dict=train_dir.valid_utt2dom_dict,
                                    valid_uid2feat=train_dir.valid_uid2feat,
                                    valid_utt2spk_dict=train_dir.valid_utt2spk_dict, verbose=verbose,
                                    save_dir=save_dir,
                                    feat_type=feat_type,
                                    transform=transform, domain=domain)

    feat_type = 'kaldi'
    if config_args['feat_format'] == 'wav':
        file_loader = read_WaveInt if config_args['feat'] == 'int' else read_WaveFloat
        feat_type = 'wav'

    train_extract_dir = KaldiExtractDataset(dir=config_args['train_test_dir'],
                                            transform=transform_V,
                                            filer_loader=file_loader, feat_type=feat_type,
                                            trials_file=config_args['train_trials'])

    return train_dir, valid_dir, train_extract_dir


def SubLoaders(train_dir, valid_dir, train_extract_dir, config_args):
    kwargs = {'num_workers': config_args['nj'],
              'pin_memory': False}  # if args.cuda else {}
    extract_kwargs = {'num_workers': 4,
                      'pin_memory': False}  # if args.cuda else {}

    min_chunk_size = int(config_args['random_chunk'][0])
    max_chunk_size = int(config_args['random_chunk'][1])
    pad_dim = 2 if config_args['feat_format'] == 'kaldi' or 'trans_fbank' in config_args else 3

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dir)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    train_paddfunc = PadCollate3d if return_domain else PadCollate

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=config_args['batch_size'],
                                               collate_fn=train_paddfunc(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / config_args['batch_size'])),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=False if 'chisquare' not in config_args else
                                                                         config_args['chisquare'],
                                                                         #  verbose=1 if torch.distributed.get_rank() == 0 else 0
                                                                         ),
                                               shuffle=config_args['shuffle'],  **kwargs)  # sampler=train_sampler,

    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(config_args['batch_size'] / 2),
                                               collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                     min_chunk_size=min_chunk_size,
                                                                     max_chunk_size=max_chunk_size,
                                                                     verbose=0
                                                                     ),
                                               shuffle=False, **kwargs)  # , sampler=valid_sampler

    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False,
                                                       **extract_kwargs)

    return train_loader, valid_loader, train_extract_loader


def Sampler_Loaders(train_dir, valid_dir, train_extract_dir, config_args):
    kwargs = {'num_workers': config_args['nj'],
              'pin_memory': False}  # if args.cuda else {}
    extract_kwargs = {'num_workers': 4,
                      'pin_memory': False}  # if args.cuda else {}

    min_chunk_size = int(config_args['random_chunk'][0])
    max_chunk_size = int(config_args['random_chunk'][1])
    pad_dim = 2 if config_args['feat_format'] == 'kaldi' else 3

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dir)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    return_score = True if 'adaptive_select' not in config_args and 'sample_ratio' in config_args and config_args['sample_ratio'] > 0 else False
    train_paddfunc = PadCollate3d if return_domain or return_score else PadCollate

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=config_args['batch_size'],
                                               collate_fn=train_paddfunc(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / config_args['batch_size'])),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=False if 'chisquare' not in config_args else
                                                                         config_args['chisquare'],
                                                                         verbose=1 if torch.distributed.get_rank() == 0 else 0
                                                                         ),
                                               sampler=train_sampler,
                                               shuffle=config_args['shuffle'],  **kwargs)  #

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dir)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(config_args['batch_size'] / 2),
                                               collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                     min_chunk_size=min_chunk_size,
                                                                     max_chunk_size=max_chunk_size,
                                                                     verbose=0
                                                                     ),
                                               sampler=valid_sampler,
                                               shuffle=False, **kwargs)  # ,

    # extract_sampler = torch.utils.data.distributed.DistributedSampler(extract_dir)
    # sampler = extract_sampler,
    # extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False,
    #                                                 sampler=extract_sampler, **extract_kwargs)

    train_extract_sampler = torch.utils.data.distributed.DistributedSampler(
        train_extract_dir)
    #
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False,
                                                       sampler=train_extract_sampler, **extract_kwargs)

    return train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler


def ScriptGenderDatasets(config_args):

    lstm_path = config_args['data_root_dir']
    train_set = config_args['train_set']
    subset    = config_args['subset']

    input_path = config_args['input_path']

    uid2spk = {}
    with open(lstm_path + '/data/{}/{}/utt2spk'.format(train_set, subset), 'r') as f:
        for l in f.readlines():
            uid, sid = l.split()
            uid2spk[uid] = sid
        
    sid2gender, uid2gender = {}, {}
    spk_file = lstm_path + '/data/{}/{}/spk2gender'.format(train_set, subset)
    if os.path.exists(spk_file):
        with open(spk_file, 'r') as f:
            for l in f.readlines():
                sid, gender = l.split()[:2]
                sid2gender[sid] = gender

        for uid in uid2spk:
            this_sid = uid2spk[uid]
            uid2gender[uid] = sid2gender[this_sid] if this_sid in sid2gender else 'null'

    spk2utt = {}
    with open(lstm_path + '/data/{}/{}/spk2utt'.format(train_set, subset), 'r') as f:
        for l in f.readlines():
            sid_uid = l.split()
            # if uid in some_data:
            spk2utt[sid_uid[0]] = sid_uid[1:]

    # load selected input uids
    data_reader = input_path + '/data.h5py'
    assert os.path.exists(data_reader), print(data_reader)

    uid_reader = input_path + '/uid_idx.json'
    assert os.path.exists(uid_reader)
    with open(uid_reader, 'r') as f:
        uididx = json.load(f)
        
    some_data = [uid for uid,idx in uididx]
    some_data.sort()

    print("Length of data: ", len(some_data))
    for uid in some_data:
        if uid2gender[uid] == 'null':
            print(uid, end=' ')
            
    all_sid = [uid2spk[uid] for uid in some_data]
    all_sid.sort()
    train_sid = set(all_sid[:-int(len(all_sid)*0.2)])

    train_uid = []
    eval_uid  = []

    num_utts = num_total = len(some_data)
    num_eval = int(np.floor(num_total*0.2))

    spk_counts = {sid:num_utts for sid in spk2utt}
    for uid in some_data:
        sid = uid2spk[uid]
        if sid in train_sid and spk_counts[sid] > num_eval:
            train_uid.append(uid) 
        else:
            eval_uid.append(uid)
            
        spk_counts[uid2spk[uid]] -= 1

    # print(len(train_uid), len(eval_uid))

    train_duration = config_args['train_duration'] #51040
    test_duration  = config_args['test_duration'] #51040

    train_dataset = GenderDataset(train_uid, uid2gender, data_reader, 
                                seg_duration=train_duration)
    eval_dataset  = GenderDataset(eval_uid, uid2gender, data_reader, if_train=False,
                                seg_duration=test_duration)

    batch_size = config_args['batch_size'] #16
    kwargs = {'num_workers': config_args['nj'], 'pin_memory': False}
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                               sampler=train_sampler, **kwargs)
    
    eval_loader  = torch.utils.data.DataLoader(eval_dataset,  batch_size=int(batch_size*2), **kwargs)

    return train_loader, eval_loader, train_sampler, train_dataset
