# NeuralGrid

Implementation of a grid-based neural network architecture in PyTorch. Both, a 2D and 3D version of a neural grid were implemented.

![activation_grid](https://kaifishr.github.io/assets/images/post10/neural_grid.png)

## 2D Neural Grid

The following image shows the neural grid's activation pattern for the different classes of the Fashion-MNIST dataset.
The input image is mapped in a first step to the grid's input dimension and is then processed from left to right using a 
simple local mapping between adjacent grid layers. The signal at the grid's output is mapped to the networks output 
using again a fully connected layer.

![activation_grid](https://kaifishr.github.io/assets/images/post10/activation_grid_end.png)

More information about this project can be found [here](https://kaifishr.github.io/2021/04/06/neural-grid.html).
