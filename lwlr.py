import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class LWLR(nn.Module):
    def __init__(self, d, kernel_fn):
        super(LWLR, self).__init__()
        self.d = d
        self.theta = np.zeros((d, 1))
        self.kernel_fn = kernel_fn

    def forward(self, x_input, X, Y):
        W = self.kernel_fn.get_weights(x_input, X)
        self.fit(X, Y, W)
        # print(self.theta.shape)
        return self.theta.T@x_input

    def fit(self, X, Y, W):
        # print(X.shape)
        # print(W.shape)
        # print(Y.shape)
        theta = np.linalg.inv(X.T@W@X)@(X.T@W@Y)
        self.theta = theta


class GaussianKernel:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # x = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        # want to output (N, N)
        N = X.shape[0]
        weights = np.exp(-1.0*(np.abs(x_input-X)**2)).flatten()
        weights = np.diag(weights)
        weights.reshape((N, N))
        # print('shape of W: ', weights.shape)
        return weights


if __name__ == '__main__':
    d = 1
    X = np.array([[1], [1], [2], [4], [4], [5], [10], [10], [11]])
    Y = np.array([[1], [2], [1], [6], [8], [7], [3], [4], [3.5]])
    plt.plot(X, Y, 'o')

    x_inputs = [3, 10.5, 15]
    y_outputs = []

    kernel = GaussianKernel(0, 1)
    model = LWLR(d, kernel)

    for x_input in x_inputs:
        y_output = model([[x_input]], X, Y)
        y_outputs.append(y_output.item())
        x_plot = np.linspace(0, 20, 30)
        # print(x_plot.shape)
        y_plot = []
        for x in x_plot:
            y_plot.append(model([[x_input]], X, Y).item())
        y_plot = np.array(y_plot)

        plt.plot(x_plot, y_plot, '-')

    plt.plot(x_inputs, y_outputs, 'x')
    plt.title('Locally Weighted Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
