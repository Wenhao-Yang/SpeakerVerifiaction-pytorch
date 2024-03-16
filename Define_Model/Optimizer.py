#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Optimizer.py
@Time: 2021/3/30 11:51
@Overview:
"""
from typing import Iterable
import numpy as np

import torch
from torch.optim import SGD
import torch.optim as optim
from torch.optim import lr_scheduler

__all__ = ["SAMSGD"]

"from https://github.com/moskomule/sam.pytorch/blob/main/sam.py"


class SAMSGD(SGD):
    """ SGD wrapped with Sharp-Aware Minimization
    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size
    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError("Invalid neighborhood size: %f" % rho)
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure().detach()

        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate

            for p in group['params']:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss


# https://github.com/neuralsyn/sam-1/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer=torch.optim.SGD, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=1e-3,
                 top_k_epoch=5, total_epochs=0,
                 config_args=None,
                 train_lengths=None,
                 verbose=True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.verbose = verbose
        self.train_lengths = train_lengths
        self.config_args = {}
        if config_args != None:
            self.config_args['scheduler'] = config_args['scheduler']
        
        self.total_epochs = total_epochs
        self.best_epoch = 0
        self.best_loss = None
        self.early_stop = False
        self.top_k_epoch = top_k_epoch
        self.top_lossepochs = []
        self.learningrates = []

    def __call__(self, val_loss, epoch, lr=None):
        self.top_lossepochs.append([val_loss, epoch])
        if lr != None:
            self.learningrates.append(lr)

        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_epoch = epoch

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")

            if self.counter >= self.patience:
                print('INFO: Early stopping, top-k epochs: ',
                      self.top_k())
                self.early_stop = True
        
        if 'scheduler' in self.config_args and self.config_args['scheduler'] != 'cyclic' and \
            (isinstance(lr, float) and lr <= 0.1 ** 3 * self.learningrates[0]):
            if len(self.learningrates) > 5:
                if self.learningrates[-5] == lr[0]:
                    self.early_stop = True

                if self.learningrates[-5] > lr[0]:
                    self.early_stop = False

    def top_k(self):
        tops = torch.tensor(self.top_lossepochs)
        tops_k_epochs = tops[torch.argsort(tops[:, 0])][:self.top_k_epoch, 1].long().tolist()

        return tops_k_epochs

class TrainSave():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=1e-3, top_k=4):
        self.lr       = []
        self.loss     = []
        self.accuracy = []
        
def create_optimizer(parameters, optimizer, **kwargs):
    # setup optimizer
    # parameters = filter(lambda p: p.requires_grad, parameters)
    if optimizer == 'sgd':
        opt = optim.SGD(parameters,
                        lr=kwargs['lr'],
                        momentum=kwargs['momentum'],
                        dampening=kwargs['dampening'],
                        weight_decay=kwargs['weight_decay'],
                        nesterov=kwargs['nesterov'])

    elif optimizer == 'adam':
        opt = optim.Adam(parameters,
                         lr=kwargs['lr'],
                         weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adagrad':
        opt = optim.Adagrad(parameters,
                            lr=kwargs['lr'],
                            lr_decay=kwargs['lr_decay'],
                            weight_decay=kwargs['weight_decay'])
    elif optimizer == 'RMSprop':
        opt = optim.RMSprop(parameters,
                            lr=kwargs['lr'],
                            momentum=kwargs['momentum'],
                            weight_decay=kwargs['weight_decay'])
    elif optimizer == 'samsgd':
        opt = SAMSGD(parameters,
                     lr=kwargs['lr'],
                     momentum=kwargs['momentum'],
                     dampening=kwargs['dampening'],
                     weight_decay=kwargs['weight_decay'])

    return opt


def create_scheduler(optimizer, config_args, train_loader=None):
    milestones = config_args['milestones']
    if config_args['scheduler'] == 'exp':
        gamma = np.power(config_args['base_lr'] / config_args['lr'],
                         1 / config_args['epochs']) if config_args['gamma'] == 0 else config_args['gamma']
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif config_args['scheduler'] == 'rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config_args['patience'], min_lr=1e-5)
    elif config_args['scheduler'] == 'cyclic':
        cycle_momentum = False if config_args['optimizer'] == 'adam' else True
        if 'step_size' in config_args:
            step_size = config_args['step_size']
        else:
            step_size = len(train_loader)

            if 'coreset_percent' in config_args and config_args['coreset_percent'] > 0:
                step_size = int(step_size * config_args['coreset_percent'])

        if 'lr_list' in config_args:
            max_lr  = config_args['lr_list']
            base_lr = [config_args['base_lr']]*len(max_lr)
        else:
            base_lr = config_args['base_lr']
            max_lr  = config_args['lr']

        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr,
                                          max_lr=max_lr,
                                          step_size_up=step_size,
                                          cycle_momentum=cycle_momentum,
                                          mode='triangular2')
    else:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

    return scheduler
