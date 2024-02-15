import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math


class LWLR(nn.Module):
    def __init__(self, D, kernel_fn):
        super(LWLR, self).__init__()
        self.D = D
        self.theta = np.zeros((D, 1))
        self.linear = nn.Linear(D, 1, bias=True)
        self.kernel_fn = kernel_fn
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, X, Y):
        W = self.kernel_fn.forward(x, X)
        self.fit(X, Y, W, 100)
        return self.linear(X)

    def fit(self, X, Y, W, epochs):

        self.train()
        prev_loss = 0
        for epoch in range(epochs):

            self.optimizer.zero_grad()
            l = self.loss(X, Y, W)

            l.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.final_epoch += 1
                print('Epoch %d/%d, Avg Loss (Risk): %.4f' % (epoch, epochs, l))

        print('*********************** Done training all epochs ***********************')

    def loss(self, X, Y, W):
        res = np.multiply(W, np.sum(Y-self.linear(X)))
        res = np.sum(res)
        return res


class GaussianKernel:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def forward(self, x_input, X):
        # x_input = (1, D), x_input is the x value for which I want to predict for
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        return math.exp(-1.0*(np.abs(x_input-X)**2))

