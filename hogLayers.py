from torch import Tensor
import torch.nn as nn

class GX_layer(nn.Module):
    def __init__(self, device='cpu', channels=3):
        super(GX_layer, self).__init__()
        self.channels = channels
        x_kernel = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        x_kernel = Tensor(x_kernel)

        x_kernel = x_kernel.view(1, 1, *x_kernel.size())
        x_kernel = x_kernel.repeat(self.channels, *[1] * (x_kernel.dim() - 1))
        x_kernel.to(device)
        self.filter = nn.Conv2d(channels, channels, kernel_size = 3, padding=1, groups=channels, bias=False)
        self.filter.weight.data = x_kernel
        self.filter.weight.requires_grad = False

    def forward(self, img):
        return self.filter(img)

class GY_layer(nn.Module):
    def __init__(self, device='cpu', channels=3):
        super(GY_layer, self).__init__()
        self.channels = channels
        y_kernel = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        y_kernel = Tensor(y_kernel)
        y_kernel = y_kernel.view(1, 1, *y_kernel.size())
        y_kernel = y_kernel.repeat(self.channels, *[1] * (y_kernel.dim() - 1))
        y_kernel.to(device)
        self.filter = nn.Conv2d(channels, channels, kernel_size = 3, padding=1, groups=channels, bias=False)
        self.filter.weight.data = y_kernel
        self.filter.weight.requires_grad = False

    def forward(self, img):
        return self.filter(img)

