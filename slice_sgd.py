import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import copy
import math


class sSGD(Optimizer):
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
        self._depth = int(len(params) / 2)
        self._re_orbit(params=params)

    def _re_orbit(self, params=None):
        if params is None:
            group = self.param_groups[0]
            params = group['params']

        g_norms = torch.zeros(self._depth, device=params[0].device)
        norm_prod = 1.
        for idx in range(self._depth):
            weight = params[idx * 2].data
            if idx == self._depth - 1:
                g_norms[idx] = weight.norm()
                norm_prod *= g_norms[idx]
                # # test
                # params[idx * 2 +1].data.div_(g_norms[idx])

                # # upd2
                # bias = params[idx * 2 + 1].data
                # bias.div_(bias.norm())

            else:
                bias = params[idx * 2 + 1].data
                g_norms[idx] = ((weight * weight).sum() + \
                    (bias * bias).sum() / (norm_prod ** 2)).sqrt()
                norm_prod *= g_norms[idx]
                bias.div_(norm_prod)
            weight.div_(g_norms[idx])

        norm_prod = norm_prod ** (1/self._depth)

        for param in params[:-1]:
            param.data.mul_(norm_prod)

    def step(self):
        group = self.param_groups[0]
        params = group['params']

        layer_sum = torch.zeros(self._depth, device=params[0].device)
        if self._renorm == 'firstlayer':
            w1 = params[0]
            b1 = params[1]
            layer_norm = (w1 * w1).sum() + (b1 * b1).sum()

        for idx in range(self._depth):
            weight = params[idx * 2]
            layer_sum[idx] = (weight.data * weight.grad.data).sum()
            if self._renorm == 'layerwise':
                layer_norm = (weight.data * weight.data).sum()

            if idx != self._depth - 1:
                bias = params[idx * 2 + 1]
                layer_sum[idx] += (bias.data * bias.grad.data).sum()
                if self._renorm == "layerwise":
                    layer_norm += (bias.data * bias.data).sum()

            layer_sum[idx] /= layer_norm

        coeff = (layer_sum.mean() - layer_sum)

        g_norms = torch.zeros(self._depth, device=coeff.device)
        norm_prod = 1.
        for idx in range(self._depth):
            weight = params[idx * 2]
            grad_weight = weight.grad.data + coeff[idx] * weight.data
            weight.data.add_(grad_weight, alpha=-group['lr'])
            if idx == self._depth - 1:
                g_norms[idx] = weight.data.norm()
                norm_prod *= g_norms[idx]
                bias = params[idx * 2 + 1]

                # grad_bias = bias.grad.data * 2
                # correct
                grad_bias = bias.grad.data

                bias.data.add_(grad_bias, alpha=-group['lr'])
                # # test
                # bias.data.div_(g_norms[idx])
            else:
                bias = params[idx * 2 + 1]
                grad_bias = bias.grad.data + coeff[idx] * bias.data
                bias.data.add_(grad_bias, alpha=-group['lr'])
                g_norms[idx] = ((weight.data * weight.data).sum() + \
                    (bias.data * bias.data).sum() / (norm_prod ** 2)).sqrt()
                norm_prod *= g_norms[idx]
                bias.data.div_(norm_prod)
            weight.data.div_(g_norms[idx])

        norm_prod = norm_prod ** (1/self._depth)
        for param in params[:-1]:
            param.data.mul_(norm_prod)
