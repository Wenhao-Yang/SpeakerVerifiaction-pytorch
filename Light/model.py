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
import torch.nn as nn
import torch
import os
from kaldiio import WriteHelper
from pytorch_lightning import LightningModule
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss, \
    VarianceLoss, DistributeLoss, MMD_Loss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AMSoftmaxLoss, ArcSoftmaxLoss, \
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2
import Process_Data.constants as C

from TrainAndTest.common_func import create_optimizer, create_scheduler


class SpeakerLoss(nn.Module):

    def __init__(self, config_args):
        super().__init__()

        self.config_args = config_args
        iteration = 0

        ce_criterion = nn.CrossEntropyLoss()
        if config_args['loss_type'] == 'soft':
            xe_criterion = None
        elif config_args['loss_type'] == 'asoft':
            ce_criterion = AngleSoftmaxLoss(
                lambda_min=config_args['lambda_min'], lambda_max=config_args['lambda_max'])
            xe_criterion = None

        elif config_args['loss_type'] == 'center':
            xe_criterion = CenterLoss(
                num_classes=config_args['num_classes'], feat_dim=config_args['embedding_size'])
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

        elif config_args['loss_type'] == 'amsoft':
            ce_criterion = None
            xe_criterion = AMSoftmaxLoss(
                margin=config_args['margin'], s=config_args['s'])
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
        elif config_args['loss_type'] == 'ring':
            xe_criterion = RingLoss(ring=config_args['ring'])
            # args.alpha = 0.0

        elif 'arcdist' in config_args['loss_type']:
            ce_criterion = DistributeLoss(
                stat_type=config_args['stat_type'], margin=config_args['m'])
            xe_criterion = ArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                          all_iteraion=0 if 'all_iteraion' not in config_args else config_args['all_iteraion'])

        self.ce_criterion = ce_criterion
        self.xe_criterion = xe_criterion
        self.loss_ratio = config_args['loss_ratio']

    def forward(self, classfier, feats, label, batch_weight=None):
        config_args = self.config_args

        if config_args['loss_type'] in ['soft', 'asoft']:
            loss = self.ce_criterion(classfier, label)

        elif config_args['loss_type'] in ['center', 'mulcenter', 'gaussian', 'coscenter', 'variance']:
            loss_cent = self.ce_criterion(classfier, label)
            loss_xent = self.loss_ratio * self.xe_criterion(feats, label)
            other_loss += loss_xent

            loss = loss_xent + loss_cent
        elif config_args['loss_type'] == 'ring':
            loss_cent = self.ce_criterion(classfier, label)
            loss_xent = self.loss_ratio * xe_criterion(feats)

            other_loss += loss_xent
            loss = loss_xent + loss_cent
        elif config_args['loss_type'] in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', ]:
            loss = self.xe_criterion(classfier, label)

            if batch_weight != None:
                loss = loss * batch_weight
                loss = loss.mean()
                self.xe_criterion.ce.reduction = 'mean'

        elif 'arcdist' in config_args['loss_type']:
            loss_xent = self.xe_criterion(classfier, label)
            loss_cent = self.loss_ratio * self.ce_criterion(classfier, label)
            # if 'loss_lambda' in config_args and config_args['loss_lambda']:
            #     loss_cent = loss_cent * lambda_

            if batch_weight != None:
                loss_xent = loss_xent * batch_weight
                loss_xent = loss_xent.mean()
                self.xe_criterion.ce.reduction = 'mean'

            other_loss += loss_cent
            loss = loss_xent + loss_cent

        return loss


class SpeakerModule(LightningModule):

    def __init__(self, config_args) -> None:
        super().__init__()

        self.config_args = config_args
        self.encoder = config_args['embedding_model']
        self.encoder.classifier = config_args['classifier']

        self.loss = SpeakerLoss(config_args)
        self.batch_size = config_args['batch_size']

        # self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        data, label = batch

        logits, embeddings = self.encoder(data)
        # logits = self.decoder(embeddings)
        loss = self.loss(logits, embeddings, label)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.valid_total_loss = 0.
        self.valid_other_loss = 0.
        self.softmax = nn.Softmax(dim=1)

        self.valid_correct = 0.
        self.valid_total_datasize = 0.
        self.valid_batch = 0.

        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        data, label = batch

        logits, embeddings = self.encoder(data)
        # logits = self.decoder(embeddings)
        val_loss = self.loss(logits, embeddings, label)

        self.valid_total_loss += float(val_loss.item())
        predicted_one_labels = self.softmax(logits)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]
        batch_correct = (predicted_one_labels == label).sum().item()

        self.valid_correct += batch_correct
        self.valid_total_datasize += len(predicted_one_labels)
        self.valid_batch += 1
        # accuracy = logits, label

        # self.log("val_loss: {:>5.2f} val_accuracy: {}{:>5.2f}%".format(
        # val_loss, batch_correct/len(predicted_one_labels)*100))
        self.log("val_batch_loss", val_loss)

        return val_loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        valid_loss = self.valid_total_loss / self.valid_batch
        valid_accuracy = 100. * self.valid_correct / self.valid_total_datasize

        # print(valid_loss, valid_accuracy)
        self.log("val_loss", valid_loss)
        self.log("val_accuracy", valid_accuracy)
        print('val_loss: {:>8.4f} val_accuracy: {:>6.2f}%'.format(
            valid_loss, valid_accuracy))
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
        return ({'optimizer': optimizer, 'scheduler': scheduler, 'monitor': 'val_loss'},)
