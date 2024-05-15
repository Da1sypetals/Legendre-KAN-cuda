import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import legendre_ops


class LegendreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, degree):
        """
        returns: cheby
        Note: degree does not require grad
        """
        # ctx.save_for_backward(x)

        batch_size, in_feats = x.size()
        legendre = legendre_ops.forward(x, degree)

        ctx.save_for_backward(x, legendre)

        return legendre


    @staticmethod
    def backward(ctx, grad_output): 
        # print(f'{grad_output.size()=}')
        x, legendre = ctx.saved_tensors

        grad_x = legendre_ops.backward(grad_output, x, legendre)

        return grad_x, None # None for degree

















