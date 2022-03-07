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

import torch
from torch.optim import SGD

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
