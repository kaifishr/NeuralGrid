"""Program to visualize neural grid of a grid neural network.

Program loads a pre-trained PyTorch model and visualizes the neural grid's
activation pattern.

"""
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

# import torchvision
import torchvision.transforms as transforms
import yaml

from src.neural_grid_2d import GridNeuralNetwork2D
from src.neural_grid_2d import GridLayer

from src.neural_grid_3d import GridNeuralNetwork3D

# from torch.utils.data import DataLoader
from src.utils import data_generator


def plot_grid_2d(cfg, model_path):
    # Create instance of model
    model = GridNeuralNetwork2D(cfg)
    # Load pre-trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Load data
    data_dict = load_test_images(cfg)
    # Visualize grid for each class
    vis_grid_2d(model, data_dict, cfg)


def visualize_grid_3d(cfg, model_path):
    # Create instance of model
    model = GridNeuralNetwork3D(cfg)
    # Load pre-trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Load data
    data_dict = load_test_images(cfg)
    # Visualize grid for each class
    vis_grid_3d(model, data_dict, cfg)


def load_test_images(cfg):
    """Loads test images for plotting.

    Loads same number of instances of each class.

    Args:
        cfg: Configuration.
    """
    # Parameters
    n_samples = cfg["visualization"]["n_samples"]
    n_classes = cfg["data"]["n_classes"]

    # Load data
    cfg["testing"]["batch_size"] = 1
    cfg["training"]["n_workers"] = 0
    _, dataloader = data_generator(cfg)

    # Create dictionary to hold test data
    data_dict = {i: {"images": [], "labels": []} for i in range(n_classes)}

    # Extract numbers from test dataset
    initial_list = [0 for i in range(n_classes)]
    target_list = [n_samples for i in range(n_classes)]

    for image, target in dataloader:
        label = target.item()
        if initial_list[label] < n_samples:
            initial_list[label] += 1
            data_dict[label]["images"].append(image.reshape(1, -1))
            data_dict[label]["labels"].append(target)
        if initial_list == target_list:
            break

    return data_dict


activation = {}


def get_activation(name):
    """Get model's activations
    Used once after creating instance of model and before inference.
    """

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def vis_grid_2d(model, data_dict, cfg):
    """Method to plot parameters and activations of neural grid."""
    # Parameters
    n_classes = cfg["data"]["n_classes"]
    n_samples = cfg["visualization"]["n_samples"]
    cmap = cfg["visualization"]["color_map"]
    dpi = cfg["visualization"]["resolution"]
    interpolation = cfg["visualization"]["interpolation"]
    results_dir = cfg["paths"]["results"]

    ##########################################################
    # Register forward activation hooks to extract activations
    ##########################################################
    for name, module in model.named_modules():
        if isinstance(module, GridLayer):
            module.register_forward_hook(get_activation(name))

    activation_grids = {i: [] for i in range(n_classes)}
    for i in range(n_classes):
        for j in range(n_samples):
            x = data_dict[i]["images"][j]
            _ = model(x)

            activation_grid = list()
            for a in activation.values():
                activation_grid.append(a.numpy())
            activations_grid = np.array(activation_grid).T
            activation_grids[i].append(activations_grid)

    h, w = np.squeeze(np.array(activation_grids[0])).shape[1:]

    ##################
    # Plot activations
    ##################
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 4))

    for i, ax in enumerate(axes.flatten()):
        grid = np.array(activation_grids[i]).mean(axis=0)
        grid = np.squeeze(grid)
        ax.imshow(grid, cmap=cmap, interpolation=interpolation)
        ax.set_title(f"Class {i}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(results_dir + "avg_activation_grid.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 4))

    for i, ax in enumerate(axes.flatten()):
        grid = np.array(activation_grids[i]).std(axis=0)
        grid = np.squeeze(grid)
        ax.imshow(grid, cmap=cmap, interpolation=interpolation)
        ax.set_title(f"Class {i}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(results_dir + "std_activation_grid.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    ###################################
    # Extract weight and biases of grid
    ###################################
    weight_grid = list()
    bias_grid = list()

    for module in model.modules():
        if isinstance(module, GridLayer):
            weight_grid.append(module.weight.detach().numpy())
            bias_grid.append(module.bias.detach().numpy())

    weight_grid = np.array(weight_grid)[:, 1:-1].T
    bias_grid = np.array(bias_grid).T

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5))

    axes[0].imshow(weight_grid, cmap=cmap, interpolation=interpolation)
    axes[0].set_title("Weights")
    axes[1].imshow(bias_grid, cmap=cmap, interpolation=interpolation)
    axes[1].set_title("Biases")

    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(results_dir + "parameter_grid.png", dpi=dpi)
    plt.close(fig)


def vis_grid_3d(model, data_dict, cfg, visualize_class=0):
    """Method to plot parameters and activations of three-dimensional neural grid."""

    # Parameters
    n_samples = cfg["visualization"]["n_samples"]
    cmap = cfg["visualization"]["color_map"]
    dpi = cfg["visualization"]["resolution"]
    results_dir = cfg["paths"]["results"]
    interpolation = cfg["visualization"]["interpolation"]
    image_height = cfg["data"]["image_height"]
    image_width = cfg["data"]["image_width"]
    n_channels = cfg["data"]["n_channels"]

    fontsize = 4

    # Extract activations
    for name, module in model.named_modules():
        if "grid_layers" in name:
            module.register_forward_hook(get_activation(name))

    # Get random sample of defined number
    rnd_idx = np.random.randint(n_samples)
    x = data_dict[visualize_class]["images"][rnd_idx].reshape(1, n_channels, image_height, image_width)
    _ = model(x)

    activation_grid = list()
    for a in activation.values():
        activation_grid.append(a.numpy())
    activation_grid = np.squeeze(np.array(activation_grid))

    # Activations 3d (basically a 4d plot)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = np.abs(activation_grid) > 0.3
    idx = np.arange(int(np.prod(activation_grid.shape)))
    x, y, z = np.unravel_index(idx, activation_grid.shape)
    ax.scatter(
        x,
        y,
        z,
        c=activation_grid.flatten(),
        s=10.0 * mask,
        alpha=0.4,
        marker="s",
        cmap=cmap,
        linewidth=0,
    )
    plt.tight_layout()
    plt.savefig(results_dir + "activation_grid_3d_scatter.png", dpi=dpi)
    plt.close(fig)

    # Activation layers
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(activation_grid[i], cmap=cmap, interpolation=interpolation)
        ax.set_title(f"Layer {i}", fontsize=fontsize)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(results_dir + "activation_grid_3d.png", dpi=dpi)
    plt.close(fig)

    # Extract weight and biases of grid
    weight_grid = list()
    bias_grid = list()
    for name, p in model.named_parameters():
        if "neural_grid" in name:
            if "weight" in name:
                weight_grid.append(p.detach().numpy())
            if "bias" in name:
                bias_grid.append(p.detach().numpy())
    weight_grid = np.squeeze(np.array(weight_grid))
    bias_grid = np.array(bias_grid)

    # Plot weights
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(weight_grid[i], cmap=cmap, interpolation=interpolation)
        ax.set_title(f"Layer {i}", fontsize=fontsize)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(results_dir + "weights_grid_3d.png", dpi=dpi)
    plt.close(fig)

    # Plot weights
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(bias_grid[i], cmap=cmap, interpolation=interpolation)
        ax.set_title(f"Layer {i}", fontsize=fontsize)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(results_dir + "biases_grid_3d.png", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":

    # Load config
    with open("config.yml", "r") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Create folder if it does not exist
    pathlib.Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)

    # Path to model to be visualized
    model_path = "models/Jan27_13-50-37/model.pth"

    plot_grid_2d(cfg, model_path)
    # plot_grid_3d(cfg, model_path)
