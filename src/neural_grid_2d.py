"""Implements a two-dimensional neural grid"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import xavier_init
from src.utils import kaiming_init

torch.manual_seed(73274)


class GridNeuralNetwork2D(nn.Module):
    """Implements a neural networks with neural grids as layer"""

    def __init__(self, params):
        super().__init__()

        # Parameters
        image_height = params["data"]["image_height"]
        image_width = params["data"]["image_width"]
        n_channels = params["data"]["n_channels"]
        n_outputs = params["data"]["n_classes"]
        grid_height = params["grid_2d"]["height"]

        # Dense layers
        n_inputs = image_height * image_width * n_channels
        self.linear_1 = nn.Linear(n_inputs, grid_height)
        self.linear_2 = nn.Linear(grid_height, n_outputs)

        self.layer_norm = nn.LayerNorm(grid_height)

        self.neural_grid = NeuralGrid(params)
        # Very slow version of neural grid, runs only on CPU with batch_size=1
        # self.neural_grid = NeuralGrid2(params)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layer_norm(self.linear_1(x))
        x = self.neural_grid(x)
        x = self.linear_2(x)
        return x


class NeuralGrid(nn.Module):
    """Implements a neural grid"""

    def __init__(self, params):
        super().__init__()

        grid_height = params["grid_2d"]["height"]
        grid_width = params["grid_2d"]["width"]

        # Add grid layers to a list
        self.grid_layers = nn.ModuleList()
        for _ in range(grid_width):
            self.grid_layers.append(GridLayer(params))

        self.layer_norm = nn.LayerNorm(grid_height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for grid_layer in self.grid_layers:
            x = x + grid_layer(torch.relu(self.layer_norm(x)))
        return x


class GridLayer(nn.Module):
    """Class implements layer of a neural grid"""

    def __init__(self, params):
        super().__init__()

        grid_height = params["grid_2d"]["height"]

        self.kernel_size = 3  # kernels must be of size 2n+1 where n>0
        self.stride = 1
        self.padding = int(0.5 * (self.kernel_size - 1))

        # Trainable parameters
        # weight = xavier_init(size=(grid_height,), fan_in=self.kernel_size, fan_out=1)
        weight = kaiming_init(size=(grid_height,), fan_in=self.kernel_size, gain=2.0**0.5)
        weight = F.pad(input=weight, pad=[self.padding, self.padding], mode="constant", value=0.0)
        self.weight = nn.Parameter(weight, requires_grad=True)
        # self.weight = nn.Parameter(0.1 + 0.0 * weight, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(size=(grid_height,)), requires_grad=True)


    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # Same padding to ensure that input size equals output size
        x = F.pad(input=x_in, pad=[self.padding, self.padding], mode="constant", value=0.0)

        # Unfold activations and weights for grid operations
        x = x.unfold(dimension=1, size=self.kernel_size, step=self.stride)
        w = self.weight.unfold(dimension=0, size=self.kernel_size, step=self.stride)
        x = (w * x).sum(dim=-1) + self.bias

        return x


class NeuralGrid2(nn.Module):
    """
    Implements a naive version of a neural grid in PyTorch
    Works only on CPU and batch_size 1
    """

    def __init__(self, params):
        super().__init__()
        self.grid_width = params["grid_2d"]["width"]
        self.grid_height = params["grid_2d"]["height"]

        # Placeholder for activations
        self.a = [[torch.zeros(size=(1,)) for _ in range(self.grid_width + 1)] for _ in range(self.grid_height + 2)]

        # Trainable parameters
        w = xavier_init(size=(self.grid_height, self.grid_width), fan_in=3, fan_out=3)
        w = F.pad(input=w, pad=[0, 0, 1, 1], mode="constant", value=0.0)
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.grid_height, self.grid_width)), requires_grad=True)

        # Activation function
        self.activation_function = torch.sin

    def forward(self, x):
        x = x.reshape(self.grid_height)

        # Assign data to grid
        for i in range(self.grid_height):
            self.a[i + 1][0] = x[i]

        # Feed data through grid
        for j in range(self.grid_width):
            for i in range(self.grid_height):
                z = (
                    self.a[i - 1][j] * self.w[i - 1][j]
                    + self.a[i][j] * self.w[i][j]
                    + self.a[i + 1][j] * self.w[i + 1][j]
                    + self.b[i][j]
                )
                self.a[i + 1][j + 1] = self.activation_function(z)

        # Assign grid output to new vector
        x_out = torch.zeros(size=(self.grid_height,))
        for i in range(self.grid_height):
            x_out[i] = self.a[i + 1][-1]

        x_out = x_out.reshape(1, self.grid_height)

        return x_out
