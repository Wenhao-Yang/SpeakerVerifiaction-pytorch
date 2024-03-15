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
import torch.nn as nn
import copy
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from sklearn.cluster import KMeans
from scipy.linalg import lstsq
from scipy.optimize import nnls
import geomloss
from geomloss import SamplesLoss

def main_process():
    if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
        return True

    return False

def kmeans_select(embeddings, select_size):
    
    if len(embeddings) >= 256:
        n_clusters = 64
    if len(embeddings) >= 64:
        n_clusters = 16
    elif len(embeddings) >= 16:
        n_clusters = 4
    else:
        n_clusters = 2
        
    embeddings = embeddings/embeddings.norm(dim=1, p=2).unsqueeze(1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings.numpy())
    sub_label = kmeans.labels_
    uniq_sub_label = np.unique(sub_label)
    subselect_size = int(np.ceil(select_size / len(uniq_sub_label)))
    
    select_idx = []
    for i in uniq_sub_label:
        this_subidx = np.arange(len(sub_label))[sub_label == i]
        select_idx.extend([j for j in this_subidx[torch.randperm(len(this_subidx))[:subselect_size]]])
    
    random.shuffle(select_idx)    
    
    return torch.LongTensor(select_idx)[:select_size]
    

def stratified_sampling(score, coreset_num, stratas=50,
                        embeddings=None, stratas_select='random'):
        # print('Using stratified sampling...')
        score = torch.tensor(score).float()
        total_num = coreset_num

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num


            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(score.shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            
            if budgets[i] < pool.shape[0]:
                if stratas_select == 'random':
                    rand_index = torch.randperm(pool.shape[0])[:budgets[i]]
                elif stratas_select == 'kmeans':
                    rand_index = kmeans_select(embeddings=embeddings[mask], select_size=budgets[i])
            
            else:
                rand_index = torch.arange(pool.shape[0])
                
            selected_index += [idx.item() for idx in pool[rand_index]]

        return selected_index


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

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

    def select(self, **kwargs):
        return
    
    def load_subset(self):
        csv_path = os.path.join(self.save_dir, 'subtrain.{}.csv'.format(self.iteration))
        score_path = os.path.join(self.save_dir, 'subtrain.{}.scores.csv'.format(self.iteration))

        top_examples = None

        if self.save_dir != '' and os.path.exists(csv_path):
            try:
                sub_utts = pd.read_csv(csv_path).to_numpy().tolist()
                assert len(sub_utts[0]) == 2 and isinstance(sub_utts[0][0], int)
                top_examples = np.array([t[0] for t in sub_utts])
            except Exception as e:
                pass

        self.norm_mean = None
        if self.save_dir != '' and os.path.exists(score_path):
            try:
                scores_utts = pd.read_csv(score_path).to_numpy()
                # assert len(sub_utts[0]) == 2 and isinstance(sub_utts[0][0], int)
                # top_examples = np.array([t[0] for t in sub_utts])
                self.norm_mean = scores_utts[:, -1].astype(np.float32)

            except Exception as e:
                pass

        return top_examples


    def save_subset(self, top_examples):
        if self.save_dir != '' and main_process():
            # print(len(self.train_dir.base_utts), np.max(top_examples))
            sub_utts = [[t, self.train_dir.base_utts[t]] for t in top_examples]
            train_utts = pd.DataFrame(sub_utts, columns=['idx', 'uids'])
            train_utts.to_csv(os.path.join(self.save_dir, 'subtrain.{}.csv'.format(self.iteration)),
                              index=None)
            
            sub_scores = self.norm_mean
            train_utts = pd.DataFrame(self.train_dir.base_utts, columns=['uid', 'start', 'end'])
            train_utts['scores'] = sub_scores

            train_utts.to_csv(os.path.join(self.save_dir, 'subtrain.{}.scores.csv'.format(self.iteration)),
                              index=None)
    

class GraNd(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4, select_aug=False,
                 save_dir='', select_sample='top', stratas=50,
                 stratas_select='random',
                 model=None, balance=False, **kwargs):
        
        super(GraNd, self).__init__(train_dir, args, fraction, random_seed, save_dir)
        
        # self.epochs = epochs
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device if model != None else None
        self.select_sample = select_sample
        self.stratas = stratas
        self.stratas_select = stratas_select

        self.random_seed += torch.distributed.get_rank()

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.device = self.model.device
            self.model = self.model.module
        
        self.embedding_dim = self.model.embedding_size

    def run(self):
        # seeding
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        batch_size = self.args['batch_size'] // 2 if not self.select_aug else self.args['batch_size'] // 4
        num_classes = self.args['num_classes']

        # self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()

        embedding_dim = self.embedding_dim #get_last_layer().in_features
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
                parameters_grads = torch.autograd.grad(loss, [classfier, embedding])
                bias_parameters_grads = parameters_grads[0]
                
                if self.stratas_select == 'kmeans':
                    embedding_grads = parameters_grads[1]
                    self.embeddings[i * batch_size:min((i + 1) * batch_size, sample_num), self.cur_repeat] = embedding_grads.detach().cpu()
                    
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
        top_examples = self.load_subset()

        if top_examples == None:
            self.model = model
            self.before_run()
            
            # Initialize a matrix to save norms of each sample on idependent runs
            self.train_indx = np.arange(self.n_train)
            self.norm_matrix = torch.zeros([self.n_train, self.repeat],
                                        requires_grad=False).to(self.device)
            
            if self.stratas_select == 'kmeans':
                self.embeddings = torch.zeros([self.n_train, self.repeat, self.embedding_dim],
                                        requires_grad=False)
            else:
                self.embeddings = None
            
            for self.cur_repeat in range(self.repeat):
                self.run()
                self.random_seed = self.random_seed + 5

            norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
            
            torch.distributed.barrier()
            norm_means = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(norm_means, norm_mean)
            norm_mean = np.mean(norm_means, axis=0)
            self.norm_mean = norm_mean

            if self.select_sample == 'ccs':
                if self.embeddings != None:
                    self.embeddings = torch.mean(self.embeddings, dim=1)
                    
                top_examples = np.array(stratified_sampling(self.norm_mean, self.coreset_size,
                                                            self.stratas,
                                                            embeddings=self.embeddings,
                                                            stratas_select=self.stratas_select))
            elif self.select_sample == 'top':
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


class LossSelect(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4, select_aug=False,
                 save_dir='', select_sample='top', stratas=50,
                 stratas_select='random',
                 model=None, balance=False, **kwargs):
        
        super(LossSelect, self).__init__(train_dir, args, fraction, random_seed, save_dir)
        
        # self.epochs = epochs
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device if model != None else None
        self.select_sample = select_sample
        self.stratas = stratas
        self.stratas_select = stratas_select

        self.random_seed += torch.distributed.get_rank()

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.device = self.model.device
            self.model = self.model.module
            
        self.embedding_dim = self.model.embedding_size

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

        # embedding_dim = self.embedding_dim #get_last_layer().in_features
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
                
                if self.select_aug:
                    loss = loss.reshape(len(self.args['augment_pipeline']), -1)
                    loss = loss.mean(dim=0)
                
                self.norm_matrix[i * batch_size:min((i + 1) * batch_size, sample_num),
                self.cur_repeat] = loss
        
        self.model.loss.reduction = previous_reduction    


    def select(self, model, **kwargs):
        top_examples = self.load_subset()

        if top_examples == None:
            self.model = model
            self.before_run()
            
            # Initialize a matrix to save norms of each sample on idependent runs
            self.train_indx = np.arange(self.n_train)
            self.norm_matrix = torch.zeros([self.n_train, self.repeat],
                                        requires_grad=False).to(self.device)
            
            if self.stratas_select == 'kmeans':
                self.embeddings = torch.zeros([self.n_train, self.repeat, self.embedding_dim],
                                        requires_grad=False)
            else:
                self.embeddings = None
            
            for self.cur_repeat in range(self.repeat):
                self.run()
                self.random_seed = self.random_seed + 5

            norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
            
            torch.distributed.barrier()
            norm_means = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(norm_means, norm_mean)
            norm_mean = np.mean(norm_means, axis=0)
            self.norm_mean = norm_mean

            if self.select_sample == 'ccs':
                if self.embeddings != None:
                    self.embeddings = torch.mean(self.embeddings, dim=1)
                    
                top_examples = np.array(stratified_sampling(self.norm_mean, self.coreset_size,
                                                            self.stratas,
                                                            embeddings=self.embeddings,
                                                            stratas_select=self.stratas_select))
            elif self.select_sample == 'top':
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

class GradMatch(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5, 
                 random_seed=None, repeat=4, 
                 save_dir='',
                 epochs=200, model=None, select_aug=False,
                 balance=True, dst_val=None, lam: float = 1., **kwargs):
        
        super(GradMatch, self).__init__(train_dir, args, fraction, random_seed)
        self.balance = balance
        self.dst_val = dst_val

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def orthogonal_matching_pursuit(self, A, b, budget: int, lam: float = 1.):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to:
        https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          budget: selection budget
          lam: regularization coef. for the final output vector
        Returns:
           vector of length n
        '''
        with torch.no_grad():
            d, n = A.shape
            if budget <= 0:
                budget = 0
            elif budget > n:
                budget = n

            x = np.zeros(n, dtype=np.float32)
            resid = b.clone()
            indices = []
            boolean_mask = torch.ones(n, dtype=bool, device="cuda")
            all_idx = torch.arange(n, device='cuda')

            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                projections = torch.matmul(A.T, resid)
                index = torch.argmax(projections[boolean_mask])
                index = all_idx[boolean_mask][index]

                indices.append(index.item())
                boolean_mask[index] = False

                if indices.__len__() == 1:
                    A_i = A[:, index]
                    x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                    A_i = A[:, index].view(1, -1)
                else:
                    A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device="cuda")
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
            if budget > 1:
                x_i = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())[0]
                x[indices] = x_i
            elif budget == 1:
                x[indices[0]] = 1.
        return x

    def orthogonal_matching_pursuit_np(self, A, b, budget: int, lam: float = 1.):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to:
        https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          budget: selection budget
          lam: regularization coef. for the final output vector
        Returns:
           vector of length n
        '''
        d, n = A.shape
        if budget <= 0:
            budget = 0
        elif budget > n:
            budget = n

        x = np.zeros(n, dtype=np.float32)
        resid = np.copy(b)
        indices = []
        boolean_mask = np.ones(n, dtype=bool)
        all_idx = np.arange(n)

        for i in range(budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            projections = A.T.dot(resid)
            index = np.argmax(projections[boolean_mask])
            index = all_idx[boolean_mask][index]

            indices.append(index.item())
            boolean_mask[index] = False

            if indices.__len__() == 1:
                A_i = A[:, index]
                x_i = projections[index] / A_i.T.dot(A_i)
            else:
                A_i = np.vstack([A_i, A[:, index]])
                x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            resid = b - A_i.T.dot(x_i)
        if budget > 1:
            x_i = nnls(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            x[indices] = x_i
        elif budget == 1:
            x[indices[0]] = 1.
        return x

    def calc_gradient(self, index=None, val=False):
        self.model.eval()
        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.train_dir if index is None else torch.utils.data.Subset(self.train_dir, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)],
                                requires_grad=False, device=self.args.device)

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device)).requires_grad_(True)
            loss = self.criterion(outputs, targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs, retain_graph=True)[0].cpu()
                weight_parameters_grads = self.model.embedding_recorder.embedding.cpu().view(batch_num, 1,
                                                    self.embedding_dim).repeat(1,self.args.num_classes,1) *\
                                                    bias_parameters_grads.view(batch_num, self.args.num_classes,
                                                    1).repeat(1, 1, self.embedding_dim)
                gradients[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] =\
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)

        return gradients

    def select(self, model, **kwargs):
        self.model = model
        self.before_run()

        self.model.no_grad = True
        with self.model.embedding_recorder:
            if self.dst_val is not None:
                val_num = len(self.dst_val.targets)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                weights = np.array([], dtype=np.float32)
                for c in range(self.args.num_classes):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    cur_gradients = self.calc_gradient(class_index)
                    if self.dst_val is not None:
                        # Also calculate gradients of the validation set.
                        val_class_index = np.arange(val_num)[self.dst_val.targets == c]
                        cur_val_gradients = torch.mean(self.calc_gradient(val_class_index, val=True), dim=0)
                    else:
                        cur_val_gradients = torch.mean(cur_gradients, dim=0)
                    if self.args.device == "cpu":
                        # Compute OMP on numpy
                        cur_weights = self.orthogonal_matching_pursuit_np(cur_gradients.numpy().T,
                                                                          cur_val_gradients.numpy(),
                                                                        budget=round(len(class_index) * self.fraction))
                    else:
                        cur_weights = self.orthogonal_matching_pursuit(cur_gradients.to(self.args.device).T,
                                                                       cur_val_gradients.to(self.args.device),
                                                                       budget=round(len(class_index) * self.fraction))
                    selection_result = np.append(selection_result, class_index[np.nonzero(cur_weights)[0]])
                    weights = np.append(weights, cur_weights[np.nonzero(cur_weights)[0]])
            else:
                cur_gradients = self.calc_gradient()
                if self.dst_val is not None:
                    # Also calculate gradients of the validation set.
                    cur_val_gradients = torch.mean(self.calc_gradient(val=True), dim=0)
                else:
                    cur_val_gradients = torch.mean(cur_gradients, dim=0)
                if self.args.device == "cpu":
                    # Compute OMP on numpy
                    cur_weights = self.orthogonal_matching_pursuit_np(cur_gradients.numpy().T,
                                                                      cur_val_gradients.numpy(),
                                                                      budget=self.coreset_size)
                else:
                    cur_weights = self.orthogonal_matching_pursuit(cur_gradients.T, cur_val_gradients,
                                                                   budget=self.coreset_size)
                selection_result = np.nonzero(cur_weights)[0]
                weights = cur_weights[selection_result]
        self.model.no_grad = False
        return {"indices": selection_result, "weights": weights}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result


class RandomSelect(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=4, save_dir='',
                 model=None, balance=False, **kwargs):
        
        super(RandomSelect, self).__init__(train_dir, args, fraction, random_seed, save_dir)
        
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.random_seed += torch.distributed.get_rank()
        
    def select(self, model, **kwargs):
        
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
        # print(norm_mean.shape)

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
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)
        self.iteration += 1

        return subtrain_dir

def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """ 
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / (a.norm(dim=2)[:, :, None]+1e-12)
            y_norm = b / (b.norm(dim=2)[:, :, None]+1e-12)
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / (a.norm(dim=1)[:, None]+1e-12)
            y_norm = b / (b.norm(dim=1)[:, None]+1e-12)
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        # M = pow(M, p)
        return M
    
def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            # if i % print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    
    return index[select_result]

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

class kCenterGreedy(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5, random_seed=1234, epochs=0,
                 repeat=4, save_dir='',
                 model=None, balance: bool = False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = True, **kwargs):
        
        super(kCenterGreedy, self).__init__(train_dir, args, fraction, random_seed,
                                        save_dir)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
            
        self.already_selected = np.array(already_selected)
        self.min_distances = None
        self.metric_ytpe = metric

        if metric in set(["cosine", "euclidean"]):
            self.metric = cost_func
        # elif callable(metric):
        #     self.metric = metric
        # else:
        #     self.metric = euclidean_dist
        #     self.run = lambda : self.finish_run()
        #     def _construct_matrix(index=None):
        #         data_loader = torch.utils.data.DataLoader(
        #             self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
        #             batch_size=self.n_train if index is None else len(index),
        #             num_workers=self.args.workers)
        #         inputs, _ = next(iter(data_loader))
        #         return inputs.flatten(1).requires_grad_(False).to(self.args.device)
        #     self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        batch_size = self.args['batch_size'] // 2 if not self.select_aug else self.args['batch_size'] // 4

        with torch.no_grad():
            matrix = []
            batch_loader = torch.utils.data.DataLoader(self.train_dir if index is None else
                                torch.utils.data.Subset(self.train_dir, index),
                                batch_size=batch_size,
                                num_workers=self.args['nj'])

            if torch.distributed.get_rank() == 0 :
                pbar = tqdm(enumerate(batch_loader), total=len(batch_loader), ncols=50)
            else:
                pbar = enumerate(batch_loader)

            for i, (data, label) in pbar:
                _, embedding = self.model(data.to(self.device))
                matrix.append(embedding)

        return torch.cat(matrix, dim=0)

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.device = self.model.device
            self.model = self.model.module
        
        self.embedding_dim = self.model.embedding_size
        # self.emb_dim = self.model.get_last_layer().in_features

    def select(self, model, **kwargs):
        self.model = model
        self.before_run()

        # self.run()
        # if self.balance:
        #     selection_result = np.array([], dtype=np.int32)
        #     for c in range(self.args.num_classes):
        #         class_index = np.arange(self.n_train)[self.dst_train.targets == c]

        #         selection_result = np.append(selection_result, k_center_greedy(self.construct_matrix(class_index),
        #                                                                        budget=round(
        #                                                                            self.fraction * len(class_index)),
        #                                                                        metric=self.metric,
        #                                                                        device=self.args.device,
        #                                                                        random_seed=self.random_seed,
        #                                                                        index=class_index,
        #                                                                        already_selected=self.already_selected[
        #                                                                            np.in1d(self.already_selected,
        #                                                                                    class_index)],
        #                                                                        print_freq=self.args.print_freq))
        # else:
        matrix = self.construct_matrix()
        # del self.model_optimizer
        # del self.model
        selection_result = k_center_greedy(matrix, budget=self.coreset_size,
                                            metric=self.metric, device=self.device,
                                            random_seed=self.random_seed,
                                            already_selected=self.already_selected, print_freq=100)
        
        return {"indices": selection_result}


class Dist_loss(nn.Module):
    def __init__(self, w, metric='cosine'):
        super(Dist_loss, self).__init__()
        w = w/w.mean()
        self.w = nn.Parameter(w)
        self.loss = SamplesLoss("sinkhorn", p=2, blur=0.05,
                                cost=lambda a, b: cost_func(a, b, p=2, metric=metric))
                                # SamplesLoss("sinkhorn", p=2, blur=0.1)
    
    def get_w(self):
        w = torch.clamp(self.w, min=0.0, max=2, out=None)
        w = w/w.mean()
        return w
    def forward(self, x1, x2):
        # w =  torch.clamp(self.w, min=0, max=2, out=None)
        # w = w / w.mean()
        w = self.get_w()
        
        return self.loss(w*x1, x2)
    

class OTSelect(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5,
                 random_seed=1234, repeat=1, select_aug=False,
                 save_dir='', scores='max',
                 model=None, balance=False, **kwargs):
        
        super(OTSelect, self).__init__(train_dir, args, fraction, random_seed, save_dir)
        
        # self.epochs = epochs
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device if model != None else 'cpu'
        self.scores = scores     

        self.random_seed += torch.distributed.get_rank()

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

        self.device = next(self.model.parameters()).device
        # self.model.device

    def run(self):
        # seeding
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        batch_size = self.args['batch_size'] if not self.select_aug else self.args['batch_size'] // 2
        num_classes = self.args['num_classes']
        optim_embeddings = self.args['optim_embeddings'] if 'optim_embeddings' in self.args else True

        # self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()

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
            loss, _ = self.model.loss(classfier, label.to(self.device),
                                        batch_weight=None, other=True)

            with torch.no_grad():
                if optim_embeddings:
                    embedding_vector = embedding
                else:
                    embedding_vector = torch.autograd.grad(loss, embedding)[0]

                # embedding_vector = embedding
                if self.select_aug:
                    embedding_vector = embedding_vector.reshape(len(self.args['augment_pipeline']), -1, embedding_vector.shape[-1])
                    embedding_vector = embedding_vector.mean(dim=0)

                self.embeddings[i * batch_size:min((i + 1) * batch_size, sample_num)] += embedding_vector
                
            # if i > 1300: 
            #     break
        # self.model.train()
        # self.model.embedding_recorder.record_embedding = False

    def select(self, model, **kwargs):
        top_examples = self.load_subset()

        self.train_indx = np.arange(self.n_train)
        if not isinstance(top_examples, np.ndarray):
            if not isinstance(self.norm_mean, np.ndarray):
                self.model = model
                self.before_run()
                embedding_dim = self.model.embedding_size #get_last_layer().in_features
                
                # Initialize a matrix to save norms of each sample on idependent runs
                
                self.embeddings = torch.zeros([self.n_train, embedding_dim],
                                            requires_grad=False).to(self.device)
                
                self.norm_matrix = torch.zeros([self.n_train, self.repeat],
                                            requires_grad=False).to(self.device)

                # for self.cur_repeat in range(self.repeat):
                self.run()
                # self.random_seed = self.random_seed + 5

                # norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
                embedding = self.embeddings.detach().cpu() / self.repeat

                torch.distributed.barrier()
                embeddings = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(embeddings, embedding)
                embedding = torch.stack(embeddings, dim=0).mean(dim=0)

                # OT_solver = SamplesLoss("sinkhorn", p=2, blur=0.1)
                # lr = 1
                batch_size = self.args['batch_size'] * 6 if not self.select_aug else self.args['batch_size'] * 2
                metric = self.args['metric'] if 'metric' in self.args else 'cosine'
                optimizer_time = self.args['optim_times'] if 'optim_times' in self.args else 1.5
                step = 20

                wss = []
                for self.cur_repeat in range(self.repeat):
                    np.random.seed(self.random_seed)
                    self.random_seed = self.random_seed - 5
                    
                    ws =  torch.ones(self.n_train, 1)
                    # losses = []
                    total_set = set(np.arange(self.n_train))

                    if torch.distributed.get_rank() == 0 :
                        pbar = tqdm(range(int(self.n_train/batch_size*optimizer_time)), ncols=50)
                    else:
                        pbar = range(int(self.n_train/batch_size*optimizer_time))

                    for i in pbar:
                        # select_ids = np.random.choice(np.arange(len(xvectors)), size=64)

                        if batch_size*i <= self.n_train:
                            select_ids = (np.arange(batch_size) + batch_size*i) % self.n_train
                        else:
                        # p = ws.squeeze().exp().numpy() 
                        # p /= p.sum()
                        # select_ids = np.random.choice(np.arange(len(xvectors)), p=p, size=64)
                            select_ids = np.random.choice(np.arange(self.n_train), size=batch_size, replace=False)

                        s_xvectors = embedding[select_ids]

                        x1 = torch.tensor(s_xvectors).to(self.device)
                        # w = nn.Parameter(ws[select_ids]).to(self.device)
                        # w = w.abs()
                        # w = w/w.mean()
                        dloss = Dist_loss(ws[select_ids], metric=metric).to(x1.device)

                        other_set = np.array(list(total_set.difference(set(select_ids))))
                        other_set = np.random.choice(other_set, size=int(batch_size*4), replace=False)

                        x2 = torch.tensor(embedding[other_set]).to(self.device)
                        # x2 = torch.tensor(xvectors).type(dtype)
                        opt = torch.optim.Adam(dloss.parameters(), lr=1, weight_decay=0)
                        # opt = torch.optim.SGD(dloss.parameters(), lr=0.2, weight_decay=0)
                        # OT_solver = SamplesLoss("sinkhorn", p=2, blur=0.05,
                        #                         cost=lambda a, b: cost_func(a, b, p=2, metric='cosine'))
                        # loss = []
                        for i in range(step):
                            # L_αβ = OT_solver(w*x1, x2)
                            # # L_αβ.backward()
                            # [g] = torch.autograd.grad(L_αβ, [w])
                            # w.data -= lr * g
                            # w = w.abs()
                            # w = w/w.mean()
                            L_αβ = dloss(x1, x2)
                            L_αβ.backward()
                            opt.step()
                            opt.zero_grad()

                            # loss.append(float(L_αβ.item()))
                        # losses.append(np.mean(loss))
                        # if (i+1) % 50 == 0:
                        #     break
                        w = dloss.get_w().data.cpu()
                        ws[select_ids] = w #dloss.w.data.abs().cpu()

                    wss.append(ws)
                    
                ws = torch.stack(wss, dim=0).mean(dim=0)

                torch.distributed.barrier()
                wss = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(wss, ws)
                ws = torch.stack(wss, dim=0).mean(dim=0)

                self.norm_mean = ws.squeeze().cpu().numpy()

            # else:
            #     print('Load scores from files ... ')

        noise_size = self.args['noise_size'] if 'noise_size' in self.args else 0
        
        if not self.balance:
            noise_size = int(noise_size * len(self.train_indx))
            if self.scores == 'max':
                # print(self.norm_mean)
                top_examples = self.train_indx[np.argsort(self.norm_mean)[::-1][noise_size:(noise_size+self.coreset_size)]]
            elif self.scores == 'min':
                top_examples = self.train_indx[np.argsort(self.norm_mean)[noise_size:(noise_size+self.coreset_size)]]
        else:
            top_examples = np.array([], dtype=np.int64)
            uids = [utts[0] for utts in self.train_dir.base_utts]
            sids = [self.train_dir.utt2spk_dict[uid] for uid in uids]
            label = np.array([self.train_dir.spk_to_idx[sid] for sid in sids])
            
            for c in range(self.num_classes):
                c_indx = self.train_indx[label == c]
                c_noise_size = int(noise_size * len(c_indx))
                budget = round(self.fraction * len(c_indx))
                if self.scores == 'max':
                    top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][c_noise_size:(c_noise_size+budget)]])
                elif self.scores == 'min':
                    top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[c_noise_size:(c_noise_size+budget)]])
        
        # print(top_examples.shape)
        self.save_subset(top_examples)
        # subtrain_dir = copy.deepcopy(self.train_dir)
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)
        self.iteration += 1

        return subtrain_dir


class Forgetting(SelectSubset):
    def __init__(self, train_dir, args, fraction=0.5, random_seed=1234, repeat=1,
                 select_aug=False, save_dir='',
                 model=None, balance=False, **kwargs):

        super(Forgetting, self).__init__(train_dir, args, fraction, random_seed, save_dir)

        self.balance = balance
        self.repeat = repeat
        self.model = model
        self.repeat = repeat
        self.balance = balance
        self.select_aug = select_aug
        self.device = model.device if model != None else 'cpu'

        self.random_seed += torch.distributed.get_rank()

        self.forgetting_events = torch.zeros(self.n_train, requires_grad=False)#.to(self.args.device)
        self.last_acc = torch.zeros(self.n_train, requires_grad=False)#.to(self.args.device)

    def before_train(self):
        self.train_loss = 0.
        self.correct = 0.
        self.total = 0.

    def before_run(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.model.module

        self.device = next(self.model.parameters()).device

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):

        if epoch <= self.repeat:
            batch_inds = batch_inds.squeeze()
            with torch.no_grad():
                
                cur_acc = (outputs == targets).clone().detach().requires_grad_(False).type(torch.float32).cpu()
                # print(batch_inds.shape, cur_acc.shape)
                self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds]-cur_acc)>0.01]] += 1.
                self.last_acc[batch_inds] = cur_acc

    def select(self, **kwargs):
        self.train_indx = np.arange(self.n_train)

        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples,
                                    c_indx[np.argsort(self.forgetting_events[c_indx].cpu().numpy())[::-1][:budget]])

        self.norm_mean = self.forgetting_events.cpu().numpy()
        self.save_subset(top_examples)
        # subtrain_dir = copy.deepcopy(self.train_dir)
        subtrain_dir = torch.utils.data.Subset(self.train_dir, top_examples)
        self.iteration += 1

        return subtrain_dir