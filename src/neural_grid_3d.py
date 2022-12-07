"""Implements a three-dimensional neural grid"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import xavier_init

torch.manual_seed(73274)


class GridNeuralNetwork3D(nn.Module):
    """Implements a three dimensional grid neural network"""

    def __init__(self, cfg):
        super().__init__()

        image_height = cfg["data"]["image_height"]
        image_width = cfg["data"]["image_width"]
        n_classes = cfg["data"]["n_classes"]

        self.dense = nn.Linear(image_height * image_width, n_classes)
        self.neural_grid = NeuralGrid(cfg)

    def forward(self, x):
        x = self.neural_grid(x)
        x = x.flatten(start_dim=-3, end_dim=-1)
        x = self.dense(x)
        return x


class NeuralGrid(nn.Module):
    """Implements a three dimensional neural grid"""

    def __init__(self, cfg):
        super().__init__()

        n_layers = cfg["grid_3d"]["n_layers"]
        self.grid_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.grid_layers.append(GridLayer(cfg))

    def forward(self, x):
        for grid_layer in self.grid_layers:
            x = grid_layer(x)
        return x


class GridLayer(nn.Module):
    """Class implements layer of a three dimensional neural grid"""

    def __init__(self, cfg):
        super().__init__()

        grid_height = cfg["data"]["image_height"]
        grid_width = cfg["data"]["image_width"]

        self.image_height = cfg["data"]["image_height"]
        self.image_width = cfg["data"]["image_width"]

        self.kernel_size = (3, 3)  # kernels must be of size 2n+1 where n>0

        pad = int(0.5 * (self.kernel_size[0] - 1))
        self.padding = (pad, pad)

        # Trainable parameters
        weight = xavier_init(
            size=(grid_height, grid_width),
            fan_in=self.kernel_size[0] ** 2,
            fan_out=self.kernel_size[0] ** 2,
        )

        weight.resize_(1, 1, grid_height, grid_width)
        self.weight = nn.Parameter(data=weight, requires_grad=True)
        self.bias = nn.Parameter(
            data=torch.zeros(size=(grid_height, grid_width)), requires_grad=True
        )

        self.activation_function = torch.sin

    def forward(self, x):
        # Unfold activations and weights for grid operations
        x = F.unfold(input=x, kernel_size=self.kernel_size, padding=self.padding)
        w = F.unfold(
            input=self.weight, kernel_size=self.kernel_size, padding=self.padding
        )

        # Prepare data
        x = x.view(
            -1,
            self.kernel_size[0] * self.kernel_size[1],
            self.image_width,
            self.image_height,
        )
        w = w.view(
            -1,
            self.kernel_size[0] * self.kernel_size[1],
            self.image_width,
            self.image_height,
        )

        # Pre-activation (weighted sum + bias)
        x = (w * x).sum(dim=1, keepdim=True) + self.bias

        # Compute activations
        x = self.activation_function(x)

        return x
