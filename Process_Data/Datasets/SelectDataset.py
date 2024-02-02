#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: SelectDataset.py
@Time: 2024/01/09 16:08
@Overview: 
'''

from typing import Any
import torch
import copy
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel

def main_process():
    if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
        return True

    return False

class SelectSubset(object):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=None, save_dir='',**kwargs) -> None:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        
        self.train_dir = train_dir
        self.num_classes = len(train_dir.speakers)
        self.save_dir = save_dir
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(train_dir)
        self.coreset_size = round(self.n_train * fraction)
        self.iteration = 0

    def select(self, **kwargs):
        return
    
    def save_subset(self, top_examples):
        if self.save_dir != '' and main_process():
            sub_utts = [self.train_dir.base_utts[t] for t in top_examples]
            train_utts = pd.DataFrame(sub_utts, columns=['uid', 'start', 'end'])
            train_utts.to_csv(os.path.join(self.save_dir, 'subtrain.{}.csv'.format(self.iteration)),
                              index=None)
    

class GraNd(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4, select_aug=False,
                 save_dir='',
                 model=None, balance=False, **kwargs):
        
        super(GraNd, self).__init__(train_dir, args, fraction, random_seed, save_dir)
        
        # self.epochs = epochs
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device        

        self.random_seed += torch.distributed.get_rank()

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

    def run(self):
        # seeding
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        batch_size = self.args['batch_size'] // 2 if not self.select_aug else self.args['batch_size'] // 4
        num_classes = self.args['num_classes']

        # self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()

        embedding_dim = self.model.embedding_size #get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.train_dir, batch_size=batch_size, num_workers=self.args['nj'])
        sample_num = self.n_train

        if torch.distributed.get_rank() == 0 :
            pbar = tqdm(enumerate(batch_loader), total=len(batch_loader), ncols=50)
        else:
            pbar = enumerate(batch_loader)

        for i, (data, label) in pbar:
            # self.model_optimizer.zero_grad()
            if self.select_aug:
                with torch.no_grad():
                    wavs_aug_tot = []
                    labels_aug_tot = []
                    wavs_aug_tot.append(data.cuda()) # data_shape [batch, 1,1,time]
                    labels_aug_tot.append(label.cuda())

                    wavs = data.squeeze().cuda()
                    wav_label = label.squeeze().cuda()

                    for augment in self.args['augment_pipeline']:
                        # Apply augment
                        wavs_aug = augment(wavs, torch.tensor([1.0]*len(wavs)).cuda())
                        # Managing speed change
                        if wavs_aug.shape[1] > wavs.shape[1]:
                            wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                        else:
                            zero_sig = torch.zeros_like(wavs)
                            zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                            wavs_aug = zero_sig

                        if 'concat_augment' in self.args and self.args['concat_augment']:
                            wavs_aug_tot.append(wavs_aug.unsqueeze(1).unsqueeze(1))
                            labels_aug_tot.append(wav_label)
                        else:
                            wavs = wavs_aug
                            wavs_aug_tot[0] = wavs_aug.unsqueeze(1).unsqueeze(1)
                            labels_aug_tot[0] = wav_label
                    
                    data = torch.cat(wavs_aug_tot, dim=0)
                    label = torch.cat(labels_aug_tot)

            classfier, embedding = self.model(data.to(self.device))
            # outputs = self.model(input)
            # loss    = self.criterion(outputs.requires_grad_(True),
            #                       targets.to(self.args.device)).sum()
            loss, _ = self.model.loss(classfier, label.to(self.device),
                                        batch_weight=None, other=True)
            
            batch_num = classfier.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, classfier)[0]
                grad_norm = torch.norm(torch.cat([bias_parameters_grads, (
                        embedding.view(batch_num, 1, embedding_dim).repeat(1,
                                             num_classes, 1) * bias_parameters_grads.view(
                                             batch_num, num_classes, 1).repeat(1, 1, embedding_dim)).
                                             view(batch_num, -1)], dim=1), dim=1, p=2)
                
                if self.select_aug:
                    grad_norm = grad_norm.reshape(len(self.args['augment_pipeline']), -1)
                    grad_norm = grad_norm.mean(dim=0)

                self.norm_matrix[i * batch_size:min((i + 1) * batch_size, sample_num),
                self.cur_repeat] = grad_norm
                
            # if i > 100: 
            #     break
        # self.model.train()
        # self.model.embedding_recorder.record_embedding = False

    def select(self, model, **kwargs):
        self.model = model
        self.before_run()
        
        # Initialize a matrix to save norms of each sample on idependent runs
        self.train_indx = np.arange(self.n_train)
        self.norm_matrix = torch.zeros([self.n_train, self.repeat],
                                       requires_grad=False).to(self.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()

        torch.distributed.barrier()
        norm_means = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(norm_means, norm_mean)
        norm_mean = np.mean(norm_means, axis=0)
        self.norm_mean = norm_mean

        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            uids = [utts[0] for utts in self.train_dir.base_utts]
            sids = [self.train_dir.utt2spk_dict[uid] for uid in uids]
            label = [self.train_dir.spk_to_idx[sid] for sid in sids] 
            
            for c in range(self.num_classes):
                c_indx = self.train_indx[label == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        self.save_subset(top_examples)
        # subtrain_dir = copy.deepcopy(self.train_dir)
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)
        self.iteration += 1

        return subtrain_dir
        # return {"indices": top_examples, "scores": self.norm_mean}


class LossSelect(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4, select_aug=False,
                 save_dir='',
                 model=None, balance=False, **kwargs):
        
        super(LossSelect, self).__init__(train_dir, args, fraction, random_seed,
                                        save_dir)
        
        # self.epochs = epochs
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device        

        self.random_seed += torch.distributed.get_rank()

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

    def run(self):
        # seeding
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        batch_size = self.args['batch_size'] // 2 if not self.select_aug else self.args['batch_size'] // 4
        num_classes = self.args['num_classes']

        # self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()
        previous_reduction = self.model.loss.reduction
        self.model.loss.reduction = 'none'

        embedding_dim = self.model.embedding_size #get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.train_dir, batch_size=batch_size, num_workers=self.args['nj'])
        sample_num = self.n_train

        if torch.distributed.get_rank() == 0 :
            pbar = tqdm(enumerate(batch_loader), total=len(batch_loader), ncols=50)
        else:
            pbar = enumerate(batch_loader)

        for i, (data, label) in pbar:
            # self.model_optimizer.zero_grad()
            if self.select_aug:
                with torch.no_grad():
                    wavs_aug_tot = []
                    labels_aug_tot = []
                    wavs_aug_tot.append(data.cuda()) # data_shape [batch, 1,1,time]
                    labels_aug_tot.append(label.cuda())

                    wavs = data.squeeze().cuda()
                    wav_label = label.squeeze().cuda()

                    for augment in self.args['augment_pipeline']:
                        # Apply augment
                        wavs_aug = augment(wavs, torch.tensor([1.0]*len(wavs)).cuda())
                        # Managing speed change
                        if wavs_aug.shape[1] > wavs.shape[1]:
                            wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                        else:
                            zero_sig = torch.zeros_like(wavs)
                            zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                            wavs_aug = zero_sig

                        if 'concat_augment' in self.args and self.args['concat_augment']:
                            wavs_aug_tot.append(wavs_aug.unsqueeze(1).unsqueeze(1))
                            labels_aug_tot.append(wav_label)
                        else:
                            wavs = wavs_aug
                            wavs_aug_tot[0] = wavs_aug.unsqueeze(1).unsqueeze(1)
                            labels_aug_tot[0] = wav_label
                    
                    data = torch.cat(wavs_aug_tot, dim=0)
                    label = torch.cat(labels_aug_tot)

            with torch.no_grad():
                classfier, _ = self.model(data.to(self.device))
                loss, _ = self.model.loss(classfier, label.to(self.device),
                                            batch_weight=None, other=True)
            
                self.norm_matrix[i * batch_size:min((i + 1) * batch_size, sample_num),
                self.cur_repeat] = loss
        
        self.model.loss.reduction = previous_reduction

    def select(self, model, **kwargs):
        self.model = model
        self.before_run()
        
        # Initialize a matrix to save norms of each sample on idependent runs
        self.train_indx = np.arange(self.n_train)
        self.norm_matrix = torch.zeros([self.n_train, self.repeat],
                                       requires_grad=False).to(self.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()

        torch.distributed.barrier()
        norm_means = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(norm_means, norm_mean)
        norm_mean = np.mean(norm_means, axis=0)
        self.norm_mean = norm_mean

        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            uids = [utts[0] for utts in self.train_dir.base_utts]
            sids = [self.train_dir.utt2spk_dict[uid] for uid in uids]
            label = [self.train_dir.spk_to_idx[sid] for sid in sids] 
            
            for c in range(self.num_classes):
                c_indx = self.train_indx[label == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        self.save_subset(top_examples)
        # subtrain_dir = copy.deepcopy(self.train_dir)
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)
        self.iteration += 1

        return subtrain_dir
        # return {"indices": top_examples, "scores": self.norm_mean}


class RandomSelect(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4,
                 model=None, balance=False, **kwargs):
        
        super().__init__(train_dir, args, fraction, random_seed, model)
        
        self.model = model
        self.repeat = repeat
        self.balance = balance
        
    def select(self, **kwargs):
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.random_seed = self.random_seed + 5
        
        # Initialize a matrix to save norms of each sample on idependent runs
        self.train_indx = np.arange(self.n_train)
        norm_mean = np.random.uniform(0, 1, self.n_train)

        torch.distributed.barrier()
        norm_means = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(norm_means, norm_mean)
        norm_mean = np.mean(norm_means, axis=0)
        self.norm_mean = norm_mean
        
        # if not self.balance:
        #     random.shuffle(self.train_indx)
        #     top_examples = self.train_indx[:self.coreset_size]
        # else:
        #     top_examples = np.array([], dtype=np.int64)
        #     uids = [utts[0] for utts in self.train_dir.base_utts]
        #     sids = [self.train_dir.utt2spk_dict[uid] for uid in uids]
        #     label = [self.train_dir.spk_to_idx[sid] for sid in sids] 
        
        #     for c in range(self.num_classes):
        #         c_indx = self.train_indx[label == c]
        #         random.shuffle(c_indx)
                
        #         budget = round(self.fraction * len(c_indx))
        #         top_examples = np.append(top_examples, c_indx[:budget])
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            uids = [utts[0] for utts in self.train_dir.base_utts]
            sids = [self.train_dir.utt2spk_dict[uid] for uid in uids]
            label = [self.train_dir.spk_to_idx[sid] for sid in sids] 
            
            for c in range(self.num_classes):
                c_indx = self.train_indx[label == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])
                
        # subtrain_dir = copy.deepcopy(self.train_dir)
        # original_utts = subtrain_dir.base_utts
        # sub_utts = [original_utts[i] for i in top_examples]
        # random.shuffle(sub_utts)
        # subtrain_dir.base_utts = sub_utts
        
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)

        return subtrain_dir
    
        # return {"indices": top_examples, "scores": self.norm_mean}
