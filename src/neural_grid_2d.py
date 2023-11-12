"""Implements a two-dimensional neural grid"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_

from src.utils import xavier_init
from src.utils import kaiming_init


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
        self.linear_in = nn.Linear(n_inputs, grid_height)
        self.linear_out = nn.Linear(grid_height, n_outputs)

        self.layer_norm = nn.LayerNorm(grid_height)

        self.neural_grid = NeuralGrid(params)
        # Very slow version of neural grid, runs only on CPU with batch_size=1
        # self.neural_grid = NaiveNeuralGrid(params)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layer_norm(self.linear_in(x))
        x = self.neural_grid(x)
        x = self.linear_out(x)
        return x


class NeuralGrid(nn.Module):
    """Implements a neural grid"""

    def __init__(self, params: dict):
        super().__init__()

        grid_width = params["grid_2d"]["width"]

        blocks = []
        for _ in range(grid_width):
            blocks.append(GridBlock(params))
        self.grid_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grid_blocks(x)
        return x


class GridBlock(nn.Module):
    """Implements a neural grid"""

    def __init__(self, params: dict):
        super().__init__()
        grid_height = params["grid_2d"]["height"]
        self.grid_layer = GridLayer(params)
        # self.grid_layer = GridLayer2(params)  # Uses `unfold`-operation.
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(grid_height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x + self.layer_norm(self.gelu(self.grid_layer(x)))
        x = self.layer_norm(self.gelu(self.grid_layer(x)))
        return x


class GridLayer(nn.Module):
    """Class implements layer of a neural grid.

    Uses 1d convolution with constant kernel.
    """

    def __init__(self, params, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        grid_height = params["grid_2d"]["height"]
        self.kernel_size = kernel_size
        self.stride = stride

        weight = torch.empty(size=(1, grid_height))
        xavier_uniform_(tensor=weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

        bias = torch.empty(size=(1, grid_height))
        xavier_uniform_(tensor=bias)
        self.bias = nn.Parameter(bias, requires_grad=True)

        kernel = torch.ones(1, 1, self.kernel_size)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, dim=1)  # Add single channel dimension.
        x = x * self.weight
        x = F.conv1d(input=x, weight=self.kernel, stride=self.stride, padding="same")
        x = x + self.bias
        x = torch.squeeze(x, dim=1)
        return x


class GridLayer2(nn.Module):
    """Grid layer that uses unfolding."""

    def __init__(self, params):
        super().__init__()

        grid_height = params["grid_2d"]["height"]

        self.kernel_size = 3  # Kernels must be of size 2n+1 where n>0.
        self.stride = 1
        self.padding = int(0.5 * (self.kernel_size - 1))

        # Trainable parameters
        weight = xavier_init(size=(grid_height,), fan_in=self.kernel_size, fan_out=1)
        # weight = kaiming_init(size=(grid_height,), fan_in=self.kernel_size, gain=2.0**0.5)
        weight = F.pad(input=weight, pad=[self.padding, self.padding], mode="constant", value=0.0)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(size=(grid_height,)), requires_grad=True)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # Same padding to ensure that input size equals output size.
        x = F.pad(input=x_in, pad=[self.padding, self.padding], mode="constant", value=0.0)

        # Unfold activations and weights for grid operations.
        x = x.unfold(dimension=1, size=self.kernel_size, step=self.stride)
        w = self.weight.unfold(dimension=0, size=self.kernel_size, step=self.stride)
        x = (w * x).sum(dim=-1) + self.bias

        return x


class NaiveNeuralGrid(nn.Module):
    """
    Implements a naive version of a neural grid in PyTorch
    Works only on CPU and batch_size 1
    """

    def __init__(self, params):
        super().__init__()
        self.grid_width = params["grid_2d"]["width"]
        self.grid_height = params["grid_2d"]["height"]

        # Placeholder for activations
        self.a = [
            [torch.zeros(size=(1,)) for _ in range(self.grid_width + 1)]
            for _ in range(self.grid_height + 2)
        ]

        # Trainable parameters
        w = xavier_init(size=(self.grid_height, self.grid_width), fan_in=3, fan_out=3)
        w = F.pad(input=w, pad=[0, 0, 1, 1], mode="constant", value=0.0)
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(
            torch.zeros(size=(self.grid_height, self.grid_width)), requires_grad=True
        )

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
