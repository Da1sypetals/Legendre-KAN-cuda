import torch
import torch.nn as nn
import torch.nn.functional as F

# from .layer import LegendreKANLayer
from .layer import BetterLayer as LegendreKANLayer

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LegendreKANLayer(28*28, 256, 4)
        self.layer3 = LegendreKANLayer(256, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.layer1(x)
        x = self.layer3(x)
        return x

