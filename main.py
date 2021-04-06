"""Program to train grid-based neural networks."""

# TODO: Run naive grid implementation with new batched samples [784] -> [1, 784]
# TODO: Ask in forum how to create sparse matrices as below. Otherwise matrix vector multiplication not possible.

from src.neural_grid_2d import GridNeuralNetwork2D
from src.neural_grid_3d import GridNeuralNetwork3D
from src.utils import comp_metrics, data_generator
from torch.utils.tensorboard import SummaryWriter

import time
import torch
import pathlib
import yaml


def run_neural_grid():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    with open("config.yml", "r") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    for key, value in cfg.items():
        print("{}:".format(key))
        for k, v in value.items():
            print(k, v)
        print()

    train_loader, test_loader = data_generator(cfg)

    loss_func = torch.nn.CrossEntropyLoss()

    model = GridNeuralNetwork2D(cfg)
    # model = GridNeuralNetwork3D(cfg)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    step_size = cfg["learning_rate_scheduler"]["step_size"]
    gamma = cfg["learning_rate_scheduler"]["gamma"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    writer = SummaryWriter()

    time_stamp = writer.log_dir.split("/")[-1]
    model_path = cfg["paths"]["models"] + time_stamp + "/"
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for epoch in range(cfg["training"]["n_epochs"]):

        running_loss = 0.0
        running_accuracy = 0.0
        running_counter = 0
        running_time = 0.0

        t0 = time.time()

        model.train()

        for _, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            x = inputs
            y = labels.long()

            # Feedforward
            y_pred = model(x)

            # Loss
            loss = loss_func(y_pred, y)

            # Set old gradients to zero
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Gradient descent
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            running_accuracy += (torch.argmax(y_pred, dim=-1) == y).float().sum()
            running_counter += labels.size(0)

        running_time += (time.time() - t0)

        scheduler.step()

        # Write trainings statistics

        model.eval()

        if epoch % cfg["tracking"]["save_stats_every_n_epochs"] == 0:
            writer.add_scalar("train_loss", running_loss / running_counter, global_step=epoch)
            writer.add_scalar("train_accuracy", running_accuracy / running_counter, global_step=epoch)
            writer.add_scalar("train_step_time", running_time / running_counter, global_step=epoch)
            writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step=epoch)
            template = "epoch {} loss {:.6f} accuracy {:.5f} learning_rate {:.6f} sample_time {:.6f} epoch_time {:.2f}"
            print(template.format(epoch+1, loss.item(), running_accuracy / running_counter, scheduler.get_lr()[0],
                                  running_time / running_counter, running_time))

        if epoch % cfg["tracking"]["save_stats_every_n_epochs"] == 0:
            test_loss, test_accuracy = comp_metrics(model=model, data_loader=test_loader, device=device)
            writer.add_scalar("test_loss", test_loss, global_step=epoch)
            writer.add_scalar("test_accuracy", test_accuracy, global_step=epoch)

        if epoch % cfg["tracking"]["save_model_every_n_epochs"] == 0:
            torch.save(model.state_dict(), model_path + "model.pth")

    print("Total time:", time.strftime("%H:%M:%S", time.gmtime(time.time()-t_start)))


if __name__ == "__main__":
    run_neural_grid()
