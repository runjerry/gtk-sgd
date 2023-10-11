import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import copy
import math


class iSGD(Optimizer):
    """Injective SGD Optimizer. """

    def __init__(self, params, lr=required, 
                 momentum=0, weight_decay=0,
                 normalize_grad=False,
                 renorm=None):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        if renorm is not None:
            assert renorm in ['firstlayer', 'layerwise']
        self._renorm = renorm
        self._normalize_grad = normalize_grad
        # self._momentum = momentum
        # self._weight_decay = weight_decay

        # normalize params layerwise
        group = self.param_groups[0]
        params = group['params']
        self._depth = len(params)
        
        norm_prod = 1.
        for idx, param in enumerate(params):
            norm = param.data.norm()
            param.data.div_(norm)
            norm_prod *= norm

        norm_prod = norm_prod ** (1/self._depth)
        for param in params:
            param.data.mul_(norm_prod)

    def step(self):
        group = self.param_groups[0]
        params = group['params']

        # if self._normalize_grad:
        #     for idx, param in enumerate(params):
        #         grad_norm = torch.clamp(param.grad.data.norm(), min=1e-6)
        #         param.grad.data.div_(grad_norm)

        if self._renorm == 'firstlayer':
            w1 = params[0]
            w1_norm = (w1.data * w1.data).sum()
            if self._normalize_grad:
                layer_sum = [(p.data * p.grad.data).sum() / p.grad.data.norm() \
                    for p in params]
            else:
                layer_sum = [(p.data * p.grad.data).sum() for p in params]
            layer_sum = torch.stack(layer_sum, dim=0)
            coeff = (layer_sum.mean() - layer_sum) / w1_norm
        else:
            pdata = [p.data / p.data.norm() for p in params]
            if self._normalize_grad:
                layer_sum = [(data * p.grad.data).sum() / p.grad.data.norm() \
                    for (data, p) in zip(pdata, params)]
            else:
                layer_sum = [
                    (data * p.grad.data).sum() for (data, p) in zip(pdata, params)]
            layer_sum = torch.stack(layer_sum, dim=0)
            coeff = (layer_sum.mean() - layer_sum)

        norm_prod = 1.
        for idx, param in enumerate(params):
            if self._renorm == 'firstlayer':
                grad = param.grad.data + coeff[idx] * param.data
            else:
                grad = param.grad.data + coeff[idx] * pdata[idx] 

            param.data.add_(grad, alpha=-group['lr'])

            if self._renorm is not None:
                norm = param.data.norm()
                param.data.div_(norm)
                norm_prod *= norm

        if self._renorm is not None:
            norm_prod = norm_prod ** (1/self._depth)
            for param in params:
                param.data.mul_(norm_prod)
