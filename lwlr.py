import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class LWLR(nn.Module):
    def __init__(self, d, kernel):
        super(LWLR, self).__init__()
        self.d = d
        self.theta = np.zeros((d, 1))
        self.kernel = kernel

    def forward(self, x_input, X, Y):
        X_bias = np.ones((X.shape[0], 1))
        X = np.concatenate([X_bias, X], axis=1)

        kern_fn = kernel(mu=x_input, sigma=4)
        x_input = np.array([[x_input]])
        x_bias = np.ones((1, 1))
        x_input = np.concatenate([x_bias, x_input], axis=1)
        # print('x_input shape after bias concat: ', x_input.shape)

        W = kern_fn.get_weights(x_input, X)
        self.fit(X, Y, W)
        return self.theta.T @ x_input.T

    def get_curr_tangent_line(self, domain):
        return [self.theta.T @ np.concatenate([np.ones((1, 1)), np.array([[x]])], axis=1).T for x in domain]

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
        x_input = np.repeat(x_input, N, axis=0)  # (N, D)
        weights = np.exp(-1.0*(np.linalg.norm(x_input-X, axis=1)**2)/(2*(self.sigma**2))).flatten()
        weights = np.diag(weights)
        weights.reshape((N, N))
        # print('shape of W: ', weights.shape)
        # weights = np.eye(N)
        return weights


if __name__ == '__main__':
    d = 1
    X = np.array([[1], [1], [2], [4], [4], [5], [10], [10], [11]])
    Y = np.array([[1], [2], [1], [6], [7], [7], [3], [4], [3.5]])
    plt.plot(X, Y, 'o')

    x_inputs = [3, 10.5, 25]
    y_outputs = []

    kernel = GaussianKernel
    model = LWLR(d, kernel)

    for x_input in x_inputs:
        print("input: ", x_input)
        y_output = model(x_input, X, Y)
        print("output: ", y_output.item())
        y_outputs.append(y_output.item())
        x_plot = np.linspace(0, 35, 30)
        y_plot = np.array(model.get_curr_tangent_line(x_plot)).squeeze()
        plt.plot(x_plot, y_plot, '-')

    plt.plot(x_inputs, y_outputs, 'x')
    plt.title('Locally Weighted Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim((0, 20))
    plt.show()
