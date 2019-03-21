import torch.nn as nn
from hogLayers import GX_layer, GY_layer
from torch import sqrt, add, cat, atan

class GNet(nn.Module):
    def __init__(self, device='cpu'):
        super(GNet, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
        ).to(device)

    def forward(self, img):
        return self.base(img)

class HOGLayer(nn.Module):
    def __init__(self, device='cpu'):
        super(HOGLayer, self).__init__()
        self.GX = GX_layer(device=device)
        self.GY = GY_layer(device=device)

    def forward(self, img):
        gx = self.GX(img)
        gy = self.GY(img)
        angle = atan(gy/gx)
        mag = sqrt(add(gx.pow(2), gy.pow(2)))
        return cat((gx, gy, ang, mag), dim=1)
