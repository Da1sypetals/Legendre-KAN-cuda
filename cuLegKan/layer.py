import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops as ein

from .legendre import LegendreFunction


class LegendreKANLayer(nn.Module):
    def __init__(self, in_features, out_features, polynomial_order=3, base_activation=nn.SiLU):

        assert polynomial_order > 0, "require order > 0"

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.poly_weight = nn.Parameter(torch.randn(polynomial_order + 1, in_features, out_features))
        self.layer_norm = nn.LayerNorm(out_features)

        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')


    def forward(self, x):
        x = x.to(self.base_weight.device)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # normalize to [-1,1]. what about tanh?
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        legendre_basis = LegendreFunction.apply(x_normalized, self.polynomial_order)
        
        # print(legendre_basis.size())
        # print(self.poly_weight.size())

        y = torch.bmm(legendre_basis, self.poly_weight)
        y = torch.sum(y, dim=0)

        out = self.base_activation(self.layer_norm(base_output + y))
        
        return out



