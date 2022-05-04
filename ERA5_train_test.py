from torch.utils.data import DataLoader
from ERA_5_dataset import Dataset_ERA5
import tqdm
from tqdm import tqdm_notebook
from ERA5_lstm import LSTM_ERA5
import torch.optim
import torch.nn as nn


def train_epoch(model, optimizer, loader, loss_func, epoch, device):
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm_notebook(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device), ys.to(device)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def main():
    training_data = Dataset_ERA5("./data/df_train.csv")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_ERA5(hidden_dim=20, input_dim=1).to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    for i in range(50):
        train_epoch(model, optimizer, train_dataloader, loss_func, epoch=(i + 1), device=device)


if __name__ == "__main__":
    main()
