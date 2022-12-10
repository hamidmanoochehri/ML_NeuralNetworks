import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Check System Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for d in range(device_count):
        device_name = torch.cuda.get_device_name(d)
        print(device_name)
#
device = torch.device("cuda:0")

columns = ["variance", "skewness", "curtosis", "entropy", "label"]
# Train data
df_train = pd.read_csv("train.csv", names=columns, dtype=np.float64())
df_train = df_train.replace({"label": 0}, -1)
df_train.to_csv("train.csv", header=False, index=False)
# Test data
df_test = pd.read_csv("test.csv", names=columns, dtype=np.float64())
df_test = df_test.replace({"label": 0}, -1)
df_test.to_csv("test.csv", header=False, index=False)


class Note(Dataset):
    def __init__(self, data_path, mode):

        super(Note, self).__init__()

        # TODO
        # 1. Initialize internal data

        train = np.loadtxt(os.path.join(data_path, "train.csv"), delimiter=",")
        test = np.loadtxt(os.path.join(data_path, "test.csv"), delimiter=",")

        Xtr, ytr, Xte, yte = (
            train[:, :-1],
            train[:, -1].reshape([-1, 1]),
            test[:, :-1],
            test[:, -1].reshape([-1, 1]),
        )

        if mode == "train":
            self.X, self.y = Xtr, ytr
        elif mode == "test":
            self.X, self.y = Xte, yte
        else:
            raise Exception("Error: Invalid mode option!")

    def __getitem__(self, index):

        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        return self.X[index, :], self.y[index, :]

    def __len__(
        self,
    ):
        # Return total number of samples.
        return self.X.shape[0]


class Network(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        super(Network, self).__init__()
        layers = []
        for l in range(len(config) - 2):
            in_dim = config[l]
            out_dim = config[l + 1]

            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))
        self.net = nn.ModuleList(layers)

    def forward(self, X):
        h = X
        for layer in self.net:
            h = layer(h)
        return h


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias.data, 1)


def accuracy(loader, model):
    correct_num = 0
    samples_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device=device)
            y = y.float().to(device=device)
            scores = model(x)
            # _, predictions = scores.max(1)
            predictions = torch.sign(scores)
            correct_num += (predictions == y).sum()
            samples_num += predictions.size(0)
        return correct_num, samples_num


# Train
train = Note("", mode="train")
test = Note("", mode="test")

# Create dataloaders for training and testing
train_loader = DataLoader(dataset=train, batch_size=16, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False)

# Setting hypyterparameters
epochs = 20
lr = 1e-3
reg = 1e-5
print("Number of depths")
depth = int(input())
widths = [5, 10, 25, 50, 100]

for hidden_layers in widths:
    if depth == 9:
        config = [
            4,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            hidden_layers,
            1,
        ]
    elif depth == 5:
        config = [4, hidden_layers, hidden_layers, hidden_layers, hidden_layers, 1]
    else:
        config = [4, hidden_layers, hidden_layers, 1]
    model = Network(config).to(device)
    model = model.to("cuda:0")
    print("\nhidden_layers=", hidden_layers)
    model.apply(weight_init)

    # pass model optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    # loss function
    loss_MSE = nn.MSELoss()

    # train
    train_loss = []
    for ie in range(epochs + 1):
        loss_array_train = []
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            Xtr = X.float().to(device)
            ytr = y.float().to(device)
            pred = model(Xtr)
            loss = loss_MSE(pred, ytr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_array_train.append(loss.cpu().detach().numpy())
        train_loss.append(np.mean(loss_array_train))

    # test
    model.eval()
    test_loss = []
    for batch_idx, (X, y) in enumerate(test_loader):
        Xte = X.float().to(device)
        yte = y.float().to(device)
        pred = model(Xtr)
        loss_test = loss_MSE(pred, ytr)
        test_loss.append(loss_test.cpu().detach().numpy())
    mean_loss = np.mean(test_loss)
    print("Test loss: ", mean_loss)

    # accuracy
    num_correct, num_samples = accuracy(test_loader, model)
    accur = float(num_correct) / float(num_samples)
    print("Test accuracy =", accur)

    # plot loss vs epoch
    plt.plot(train_loss)
    plt.ylabel("Training loss")
    plt.xlabel("epochs")
    plt.show()
