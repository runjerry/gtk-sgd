import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import copy
import math


class affineSGD(Optimizer):
    """Affine GTK SGD Optimizer. """

    def __init__(self, params, lr=required,
                 momentum=0, weight_decay=0,
                 use_bias=True, fullrank=True,
                 fixed_rand_vec=False, weight_only=True, 
                 same_norm=False, norm=False, diag=False):
        assert not (fullrank and diag), (
            "fullrank and diag are incompatible with each other")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._use_bias = use_bias
        self._fullrank = fullrank
        self._weight_only = weight_only
        self._same_norm = same_norm
        self._norm = norm
        self._diag = diag

        group = self.param_groups[0]
        params = group['params']
        self._depth = len(params)
        self._fixed_rand_vec = fixed_rand_vec
        if fixed_rand_vec:
            rand_vecs = []
            for idx, param in enumerate(params):
                if self._use_bias and idx % 2 == 1 and self._weight_only:
                    rand_vecs.append(None)
                else:
                    rand_vec = torch.randn_like(param.data)
                    if self._diag:
                        rand_vec += 1
                        rand_vec = torch.clamp(rand_vec, 0.)
                    elif not self._same_norm and not self._norm:
                        rand_vec = rand_vec / rand_vec.norm()
                    rand_vecs.append(rand_vec)
            self._rand_vecs = rand_vecs

    def step(self):
        group = self.param_groups[0]
        params = group['params']

        for idx, param in enumerate(params):
            if self._use_bias and idx % 2 == 1 and self._weight_only:
                grad = param.grad.data
            else:
                if self._fixed_rand_vec:
                    rand_vec = self._rand_vecs[idx]
                else:
                    rand_vec = torch.randn_like(param.grad.data)
                    if self._diag:
                        rand_vec += 1
                        rand_vec = torch.clamp(rand_vec, 0.)
                    elif not self._same_norm and not self._norm:
                        rand_vec = rand_vec / rand_vec.norm()
                if self._diag:
                    grad = rand_vec * param.grad.data
                else:
                    grad = torch.mul(rand_vec, param.grad.data).sum() * rand_vec
                if self._fullrank:
                    grad += param.grad.data
                if self._same_norm:
                    grad = (grad / grad.norm()) * param.grad.data.norm()
                if self._norm:
                    grad = grad / grad.norm()

            param.data.add_(grad, alpha=-group['lr'])
