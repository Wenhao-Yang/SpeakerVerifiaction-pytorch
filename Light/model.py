#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: common.py
@Time: 2023/02/23 18:23
@Overview:
'''
import time
import numpy as np
import torch.nn as nn
import torch
import os
import pdb
from kaldiio import WriteHelper
from pytorch_lightning import LightningModule
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss, \
    VarianceLoss, DistributeLoss, MMD_Loss, aDCFLoss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AMSoftmaxLoss, ArcSoftmaxLoss, DAMSoftmaxLoss, \
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2, MixupLoss
import Process_Data.constants as C

from TrainAndTest.common_func import create_optimizer, create_scheduler
from Eval.eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf


class SpeakerLoss(nn.Module):

    def __init__(self, config_args):
        super().__init__()

        self.config_args = config_args
        self.lncl = True if 'lncl' in config_args and config_args['lncl'] == True else False
        iteration = 0

        ce_criterion = nn.CrossEntropyLoss()
        loss_type = set(['soft', 'asoft', 'center', 'variance', 'gaussian', 'coscenter', 'mulcenter',
                         'amsoft', 'subam',  'damsoft', 'subdam',
                         'arcsoft', 'subarc', 'minarcsoft', 'minarcsoft2', 'wasse', 'mmd', 'ring', 'arcdist',
                         'aDCF', ])

        if config_args['loss_type'] == 'soft':
            xe_criterion = None
        elif config_args['loss_type'] == 'asoft':
            ce_criterion = AngleSoftmaxLoss(
                lambda_min=config_args['lambda_min'], lambda_max=config_args['lambda_max'])
            xe_criterion = None
        elif config_args['loss_type'] == 'variance':
            xe_criterion = VarianceLoss(
                num_classes=config_args['num_classes'], feat_dim=config_args['embedding_size'])
        elif config_args['loss_type'] == 'gaussian':
            xe_criterion = GaussianLoss(
                num_classes=config_args['num_classes'], feat_dim=config_args['embedding_size'])
        elif config_args['loss_type'] == 'coscenter':
            xe_criterion = CenterCosLoss(
                num_classes=config_args['num_classes'], feat_dim=config_args['embedding_size'])
        elif config_args['loss_type'] == 'mulcenter':
            xe_criterion = MultiCenterLoss(num_classes=config_args['num_classes'], feat_dim=config_args['embedding_size'],
                                           num_center=config_args['num_center'])
        elif config_args['loss_type'] in ['amsoft', 'subam']:
            ce_criterion = None
            xe_criterion = AMSoftmaxLoss(
                margin=config_args['margin'], s=config_args['s'])
        elif config_args['loss_type'] in ['damsoft', 'subdam']:
            ce_criterion = None
            xe_criterion = DAMSoftmaxLoss(
                margin=config_args['margin'], s=config_args['s'])
        elif config_args['loss_type'] in ['aDCF']:
            ce_criterion = None
            xe_criterion = aDCFLoss(alpha=config_args['s'],
                                    beta=(1 - config_args['smooth_ratio']),
                                    gamma=config_args['smooth_ratio'],
                                    omega=config_args['margin'])

        elif config_args['loss_type'] in ['arcsoft', 'subarc']:
            ce_criterion = None
            if 'class_weight' in config_args and config_args['class_weight'] == 'cnc1':
                class_weight = torch.tensor(C.CNC1_WEIGHT)
                if len(class_weight) != config_args['num_classes']:
                    class_weight = None
            else:
                class_weight = None
            if 'dynamic_s' in config_args:
                dynamic_s = config_args['dynamic_s']
            else:
                dynamic_s = False

            all_iteraion = 0 if 'all_iteraion' not in config_args else config_args[
                'all_iteraion']
            smooth_ratio = 0 if 'smooth_ratio' not in config_args else config_args[
                'smooth_ratio']
            xe_criterion = ArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                          all_iteraion=all_iteraion, smooth_ratio=smooth_ratio,
                                          class_weight=class_weight, dynamic_s=dynamic_s)

        elif config_args['loss_type'] == 'minarcsoft':
            ce_criterion = None
            xe_criterion = MinArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                             all_iteraion=config_args['all_iteraion'])
        elif config_args['loss_type'] == 'minarcsoft2':
            ce_criterion = None
            xe_criterion = MinArcSoftmaxLoss_v2(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                                all_iteraion=config_args['all_iteraion'])
        elif config_args['loss_type'] == 'wasse':
            xe_criterion = Wasserstein_Loss(
                source_cls=config_args['source_cls'])
        elif config_args['loss_type'] == 'mmd':
            xe_criterion = MMD_Loss()
            # args.alpha = 0.0

        if 'second_loss' in config_args and config_args['second_loss'] == 'center':
            ce_criterion = CenterLoss(num_classes=config_args['num_classes'],
                                      feat_dim=config_args['embedding_size'],
                                      alpha=config_args['center_alpha'] if 'center_alpha' in config_args else 0)
        elif 'second_loss' in config_args and config_args['second_loss'] == 'ring':
            ce_criterion = RingLoss(ring=config_args['ring'])
        elif 'second_loss' in config_args and config_args['second_loss'] == 'dist':
            ce_criterion = DistributeLoss(stat_type=config_args['stat_type'],
                                          margin=config_args['m'])

        self.softmax = nn.Softmax(dim=1)
        self.ce_criterion = ce_criterion
        self.xe_criterion = xe_criterion
        self.loss_ratio = config_args['loss_ratio']

    def forward(self, classfier, feats, label, batch_weight=None, epoch=0,
                half_data=0, lamda_beta=0):

        config_args = self.config_args
        other_loss = 0.

        if config_args['loss_type'] in ['soft', 'asoft']:
            loss = self.ce_criterion(classfier, label)
        elif config_args['loss_type'] in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', ]:
            if isinstance(self.xe_criterion, MixupLoss):
                loss = self.xe_criterion(
                    classfier, label, half_batch_size=half_data, lamda_beta=lamda_beta)
            else:
                loss = self.xe_criterion(classfier, label)

            if batch_weight != None:
                loss = loss * batch_weight
                loss = loss.mean()
                self.xe_criterion.ce.reduction = 'mean'

            if self.ce_criterion != None:
                loss_cent = self.loss_ratio * self.ce_criterion(feats, label)
                other_loss += float(loss_cent)
                loss = loss + loss_cent

        # if self.lncl:
        #     predicted_labels = self.softmax(classfier.clone())
        #     predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        #     if config_args['loss_type'] in ['amsoft', 'damsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2',
        #                                     'aDCF', 'subarc', 'arcdist']:
        #         predict_loss = self.xe_criterion(
        #             classfier, predicted_one_labels)
        #     else:
        #         predict_loss = self.ce_criterion(
        #             classfier, predicted_one_labels)

        #     alpha_t = np.clip(
        #         config_args['alpha_t'] * (epoch / config_args['epochs']) ** 2, a_min=0, a_max=1)
        #     mp = predicted_labels.mean(dim=0) * predicted_labels.shape[1]

        #     loss = (1 - alpha_t) * loss + alpha_t * predict_loss + \
        #         config_args['beta'] * torch.mean(-torch.log(mp))

        return loss, other_loss


def get_trials(trials):

    trials_pairs = []

    with open(trials, 'r') as t:

        for line in t.readlines():
            pair = line.split()
            pair_true = False if pair[2] in ['nontarget', '0'] else True
            trials_pairs.append((pair[0], pair[1], pair_true))

    return trials_pairs


class SpeakerModule(LightningModule):

    def __init__(self, config_args) -> None:
        super().__init__()

        self.config_args = config_args
        # self.train_dir = train_dir
        self.encoder = config_args['embedding_model']
        self.encoder.classifier = config_args['classifier']
        self.softmax = nn.Softmax(dim=1)

        self.loss = SpeakerLoss(config_args)
        self.batch_size = config_args['batch_size']
        self.test_trials = get_trials(config_args['train_trials_path'])
        # self.optimizer = optimizer

    def on_train_epoch_start(self) -> None:
        self.train_accuracy = []
        self.train_loss = []

        self.total_forward = 0
        self.total_backward = 0
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        torch.cuda.empty_cache()

        start = time.time()
        data, label = batch

        logits, embeddings = self.encoder(data)
        loss, other_loss = self.loss(logits, embeddings, label)

        predicted_one_labels = self.softmax(logits)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]
        batch_correct = (predicted_one_labels == label).sum().item()

        train_batch_accuracy = 100. * batch_correct / len(predicted_one_labels)
        self.train_accuracy.append(train_batch_accuracy)
        self.train_loss.append(float(loss))

        self.log("train_batch_loss", float(loss))
        self.log("train_batch_accu", train_batch_accuracy)

        self.stop_time = time.time()
        self.print('forward:, ', self.stop_time - start)

        return loss

    def training_step_end(self, step_output):
        self.print('backward:, ', time.time() - self.stop_time)

        return super().training_step_end(step_output)

    # def on_train_epoch_end(self, outputs) -> None:
    #     # pdb.set_trace()
    #     # print(self.current_epoch)
    #     self.print("Epoch {:>2d} Loss: {:>7.4f} Accuracy: {:>6.2f}%".format(
    #         self.current_epoch, np.mean(self.train_loss), np.mean(self.train_accuracy)))
    #     # self.train_dir.__shuffle__()
    #     return super().on_train_epoch_end(outputs)

    def training_epoch_end(self, outputs) -> None:

        self.print("Epoch {:>2d} Loss: {:>7.4f} Accuracy: {:>6.2f}%".format(
            self.current_epoch, np.mean(self.train_loss), np.mean(self.train_accuracy)))
        return super().training_epoch_end(outputs)

    def on_validation_epoch_start(self) -> None:
        self.valid_total_loss = 0.
        self.valid_other_loss = 0.

        self.valid_correct = 0.
        self.valid_total_datasize = 0.
        self.valid_batch = 0.
        # self.valid_data = torch.tensor([])
        # self.valid_num_seg = [0]
        # self.valid_uid_lst = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # this is the validation loop
        data, label = batch
        if data.shape[1] != 1:
            data_shape = data.shape
            data = data.reshape(-1, 1, data_shape[2], data_shape[3])

        # self.valid_data = torch.cat((self.valid_data, data), dim=0)
        # self.valid_num_seg.append(self.valid_num_seg[-1] + len(data))
        # self.valid_uid_lst.append(label[0])
        # if self.valid_data.shape[0] >= self.batch_size or batch_idx+1 == len(self.val_dataloader[0]):
        #     logits, embeddings = self.encoder(self.valid_data)
        #     label = self.valid_uid_lst
        #     return embeddings, label
        logits, embeddings = self.encoder(data)
        # logits = self.decoder(embeddings)
        # ipdb.set_trace()
        if isinstance(label[0], str):
            return embeddings, label
        else:
            val_loss = self.loss(logits, embeddings, label)
            self.valid_total_loss += float(val_loss.item())
            predicted_one_labels = self.softmax(logits)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]
            batch_correct = (predicted_one_labels == label).sum().item()

            self.valid_correct += batch_correct
            self.valid_total_datasize += len(predicted_one_labels)
            self.valid_batch += 1

            # self.log("val_batch_loss", val_loss)

            return val_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # pdb.set_trace()
        for dataloader_outs in outputs:
            # pdb.set_trace()
            # dataloader_outs = dataloader_output_result  # .dataloader_i_outputs
            if isinstance(dataloader_outs[0], tuple):
                uid2embedding = {
                    uid[0]: embedding for embedding, uid in dataloader_outs}
                distances = []
                labels = []

                for a_uid, b_uid, l in self.test_trials:
                    try:
                        a = uid2embedding[a_uid]
                        b = uid2embedding[b_uid]
                    except Exception as e:
                        continue

                    a_norm = a/a.norm(2)
                    b_norm = b/b.norm(2)

                    distances.append(float(a_norm.matmul(b_norm.T).mean()))
                    labels.append(l)

                eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels,
                                                                  cos=True, re_thre=True)
                mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(
                    distances, labels)
                # pdb.set_trace()
                self.log("val_eer", eer*100,  sync_dist=True)
                self.log("val_mindcf_01", mindcf_01,  sync_dist=True)
                self.log("val_mindcf_001", mindcf_001,  sync_dist=True)

            else:
                # self.log("val_accuracy", valid_accuracy)
                valid_loss = self.valid_total_loss / self.valid_batch
                valid_accuracy = 100. * self.valid_correct / self.valid_total_datasize
                # print(valid_loss, valid_accuracy)
                self.log("val_loss", valid_loss)
                self.log("val_accuracy", valid_accuracy)
                # print('val_loss: {:>8.4f} val_accuracy: {:>6.2f}%'.format(
                #     valid_loss, valid_accuracy))
        return super().validation_epoch_end(outputs)

    def on_test_epoch_start(self) -> None:
        self.test_xvector_dir = "%s/train/epoch_%s" % (
            self.xvector_dir, self.current_epoch)
        self.test_uid2vectors = []

        return super().on_test_epoch_start()

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.test_input == 'fix':
            self.this_test_data = torch.tensor([])
            self.this_test_seg = [0]
            self.this_test_uids = []

        # elif self.test_input == 'var':
        #     max_lenght = 10 * C.NUM_FRAMES_SPECT
        #     if self.feat_type == 'wav':
        #         max_lenght *= 160

        return super().on_test_batch_start(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        data, uid = batch
        vec_shape = data.shape

        if vec_shape[1] != 1:
            data = data.reshape(
                vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        self.this_test_data = torch.cat((self.this_test_data, data), dim=0)
        self.this_test_seg.append(self.this_test_seg[-1] + len(data))
        self.this_test_uids.append(uid[0])

        # embeddings = self.encoder(data)
        batch_size = self.batch_size
        if self.this_test_data.shape[0] >= batch_size or batch_idx + 1 == len(self.test_dataloader):
            if self.this_test_data.shape[0] > (3 * batch_size):
                i = 0
                out = []
                while i < self.this_test_data.shape[0]:
                    data_part = self.this_test_data[i:(i + batch_size)]
                    model_out = self.encoder(data_part)
                    if isinstance(model_out, tuple):
                        try:
                            _, out_part, _, _ = model_out
                        except:
                            _, out_part = model_out
                    else:
                        out_part = model_out

                    out.append(out_part)
                    i += batch_size

                out = torch.cat(out, dim=0)
            else:
                model_out = self.encoder(self.this_test_data)

                if isinstance(model_out, tuple):
                    try:
                        _, out, _, _ = model_out
                    except:
                        _, out = model_out
                else:
                    out = model_out

            out = out.data.cpu().float().numpy()
            # print(out.shape)
            if len(out.shape) == 3:
                out = out.squeeze(0)

            for i, uid in enumerate(self.this_test_uids):
                uid_vec = out[self.this_test_seg[i]:self.this_test_seg[i + 1]]
                if self.mean_vector:
                    uid_vec = uid_vec.mean(axis=0)
                self.test_uid2vectors.append((uid, uid_vec))

            self.this_test_data = torch.tensor([])
            self.this_test_seg = [0]
            self.this_test_uids = []

    def on_test_epoch_end(self) -> None:

        if torch.distributed.get_rank() == 0:
            xvector_dir = self.test_xvector_dir
            if not os.path.exists(xvector_dir):
                os.makedirs(xvector_dir)

            scp_file = xvector_dir + '/xvectors.scp'
            ark_file = xvector_dir + '/xvectors.ark'
            writer = WriteHelper('ark,scp:%s,%s' % (ark_file, scp_file))

            for uid, uid_vec in self.test_uid2vectors:
                writer(str(uid), uid_vec)

        return super().on_test_epoch_end()

    def configure_optimizers(self):
        config_args = self.config_args

        opt_kwargs = {'lr': config_args['lr'],
                      'lr_decay': config_args['lr_decay'],
                      'weight_decay': config_args['weight_decay'],
                      'dampening': config_args['dampening'],
                      'momentum': config_args['momentum'],
                      'nesterov': config_args['nesterov']}

        optimizer = create_optimizer(
            self.parameters(), config_args['optimizer'], **opt_kwargs)
        scheduler = create_scheduler(optimizer, config_args)

        # torch.optim.Adam(self.parameters(), lr=1e-3)
        # return ({'optimizer': optimizer, 'scheduler': scheduler, 'monitor': 'val_loss'},)
        return ({'optimizer': optimizer,
                 'lr_scheduler': {
                     "scheduler": scheduler,
                     "monitor": "val_loss",
                 }, })
