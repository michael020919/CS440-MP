# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256


        # TODO Define the network architecture (layers) based on these specifications.
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat_size = 16 * 15 * 15
        self.f1 = nn.Linear(self.flat_size, 64)
        self.f2 = nn.Linear(64, out_size)
        self.activation = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        x = x.reshape(x.shape[0], 3, 31, 31)
        x = self.pool(self.activation(self.conv(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.activation(self.f1(x))
        return self.f2(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    num = train_set.shape[1]
    for j in range(num):
        train = train_set[:, j]
        col_dev   = dev_set[:, j]
        mean = train.mean()
        std = train.std()
        train_set[:, j] = (train - mean) / std
        dev_set[:, j]   = (col_dev  - mean) / std

    in_size  = train_set.shape[1]
    out_size = int(train_labels.max().item()) + 1

    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(lrate = 0.002, loss_fn = loss_fn, in_size = in_size, out_size = out_size)
    dataset = get_dataset_from_arrays(train_set, train_labels)
    loader  = DataLoader(dataset, batch_size)

    losses = []
    for i in range(epochs):
        sum = 0
        count = 0
        for batch in loader:
            inputs = batch["features"]
            labels = batch["labels"]
            batch_loss = net.step(inputs, labels)
            sum += batch_loss
            count += 1
            loss = sum / count
        losses.append(loss)

    outputs = net(dev_set)
    _, indices = torch.max(outputs, dim=1)
    labels = indices.detach().numpy()
    return losses, labels, net