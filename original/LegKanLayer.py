import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

import einops as ein

class KAL_Layer(nn.Module):
    def __init__(self, in_features, out_features, polynomial_order=3, base_activation=nn.SiLU):
        super(KAL_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.poly_weight = nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1)))
        self.layer_norm = nn.LayerNorm(out_features)
        
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_legendre_polynomials(x, order):
        P0 = x.new_ones(x.shape)
        if order == 0:
            return ein.rearrange(P0, 'n -> 1 n')
        P1 = x
        legendre_polys = [P0, P1]
        for d in range(1, order):
            Pd = ((2.0 * d + 1.0) * x * legendre_polys[-1] - d * legendre_polys[-2]) / (d + 1.0)
            legendre_polys.append(Pd)
        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        x = x.to(self.base_weight.device)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # what about tanh
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
        legendre_basis = legendre_basis.view(x.size(0), -1)
        
        poly_output = F.linear(legendre_basis, self.poly_weight)
        x = self.base_activation(self.layer_norm(base_output + poly_output))
        
        return x

class KAL_Net(nn.Module):
    def __init__(self, polynomial_order=3, base_activation=nn.SiLU):
        super(KAL_Net, self).__init__()
        self.base_activation = base_activation
        
        # Explicitly define each layer with its in_features and out_features
        self.layer1 = KAL_Layer(in_features=28*28, out_features=256, polynomial_order=polynomial_order, base_activation=base_activation)
        self.layer2 = KAL_Layer(in_features=256, out_features=256, polynomial_order=polynomial_order, base_activation=base_activation)
        self.layer3 = KAL_Layer(in_features=256, out_features=10, polynomial_order=polynomial_order, base_activation=base_activation)
        
        # More layers can be added explicitly if needed

    def forward(self, x):
        # Stack the layers by explicitly calling the forward method of each layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # If more layers, continue stacking them here
        return x

# Example usage:
# net = KAL_Net(layers_hidden=[512, 256, 128], polynomial_order=3, base_activation=nn.SiLU)
# output = net(input_tensor)