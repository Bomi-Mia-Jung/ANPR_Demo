import autograd.numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class LWLR(nn.Module):
    def __init__(self, d, kernel, bandwidth, lbda):
        super(LWLR, self).__init__()
        self.d = d
        self.lbda = lbda
        self.theta = np.zeros((d, 1))
        self.kernel = kernel(sigma=bandwidth)
        self.bandwidth = bandwidth

    def forward(self, x_input, X, Y):
        X_bias = np.ones((X.shape[0], 1))
        X = np.concatenate([X_bias, X], axis=1)

        x_input = np.array([[x_input]])
        x_bias = np.ones((1, 1))
        x_input = np.concatenate([x_bias, x_input], axis=1)
        # print('x_input shape after bias concat: ', x_input.shape)

        W = self.get_weights(x_input, X)
        self.fit(X, Y, W)
        return self.theta.T @ x_input.T

    def get_weights(self, x_input, X):
        N = X.shape[0]
        W = self.kernel.get_weights(x_input, X)
        W = np.diag(W)
        W.reshape((N, N))
        return W

    def get_local_line(self, x_input, X, Y, domain):
        X_bias = np.ones((X.shape[0], 1))
        X = np.concatenate([X_bias, X], axis=1)

        x_input = np.array([[x_input]])
        x_bias = np.ones((1, 1))
        x_input = np.concatenate([x_bias, x_input], axis=1)
        # print('x_input shape after bias concat: ', x_input.shape)

        W = self.get_weights(x_input, X)
        theta = np.linalg.solve(X.T@W@X + np.diag(self.lbda*np.ones((self.d+1,))), (X.T@W@Y))
        # print(theta)
        return [theta.T @ np.concatenate([np.ones((1, 1)), np.array([[x]])], axis=1).T for x in domain]

    def get_curve(self, domain, X, Y):
        return [self.forward(x, X, Y) for x in domain]

    def fit(self, X, Y, W):
        # print(X.shape)
        # print(W.shape)
        # print(Y.shape)
        theta = np.linalg.solve(X.T@W@X + np.diag(self.lbda*np.ones((self.d+1,))), (X.T@W@Y))
        self.theta = theta
        return self.theta


class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        x_input = np.repeat(x_input, X.shape[0], axis=0)  # (N, D)
        weights = np.exp(-1.0*(np.linalg.norm(x_input-X, axis=1)**2)/(2*(self.sigma**2))).flatten()
        # print('shape of W: ', weights.shape)
        # weights = np.eye(N)
        return weights

class AnovaRBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        distances = np.linalg.norm(x_input - X, axis=1)  # Compute Euclidean distances, shape: (N,)
        weights = np.exp(-1.0 * (distances ** 2) / (2 * (self.sigma ** 2)))  # Compute Gaussian weights
        return weights


class TricubeKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        distances = np.linalg.norm(x_input - X, axis=1) / self.sigma  # Compute distances normalized by bandwidth
        mask = distances <= 1  # Mask for distances within bandwidth (sigma)
        weights = np.where(mask, (1 - distances ** 3) ** 3, 0)  # Compute Tricube weights
        return weights

class EpanechnikovKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        distances = np.linalg.norm(x_input - X, axis=1) / self.sigma  # Compute distances normalized by bandwidth
        mask = distances <= 1  # Mask for distances within bandwidth
        weights = np.where(mask, 0.75 * (1 - distances ** 2), 0)  # Compute Epanechnikov weights
        return weights


class TriangularKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        distances = np.linalg.norm(x_input - X, axis=1) / self.sigma  # Compute distances normalized by bandwidth
        weights = np.where(distances <= 1, 1 - distances, 0)  # Compute Triangular weights
        return weights


class LaplacianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        distances = np.linalg.norm(x_input - X, axis=1)  # Compute Euclidean distances, shape: (N,)
        weights = np.exp(-1.0 * distances / self.sigma)  # Compute Laplacian weights
        return weights


class UniformKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_weights(self, x_input, X):
        # x_input = (1, D)
        # X = (N, D)
        # N is the number of data samples in the training set
        # D is the dimensions of the data
        return np.ones(X.shape[0])  # Uniform weights for all data points



if __name__ == '__main__':
    d = 1
    X = np.array([[1], [1], [2], [4], [4], [5], [10], [10], [11]])
    Y = np.array([[1], [2], [1], [6], [7], [7], [3], [4], [3.5]])
    plt.plot(X, Y, 'o')

    x_inputs = [3, 10.5, 25]
    y_outputs = []

    kernel = GaussianKernel
    model = LWLR(d, kernel, 2, lbda=0)

    for x_input in x_inputs:
        # print("input: ", x_input)
        y_output = model(x_input, X, Y)
        # print("output: ", y_output.item())
        y_outputs.append(y_output.item())
        x_plot = np.linspace(0, 35, 30)

        # plot the local line being learned at the test x_input
        y_plot = np.array(model.get_local_line(x_input, X, Y, x_plot)).squeeze()
        plt.plot(x_plot, y_plot, '-', label=('locally learned line at point {}'.format(x_input)))

    plt.plot(x_inputs, y_outputs, 'x')

    # plot curve
    x_plot = np.linspace(0, 35, 50)
    y_plot = np.array(model.get_curve(x_plot, X, Y)).squeeze()
    plt.plot(x_plot, y_plot, '--')

    plt.title('Locally Weighted Linear Regression')
    plt.xlabel('x')
    plt.xlim((0, 30))
    plt.ylabel('y')
    plt.ylim((0, 20))
    plt.legend(loc='best')
    plt.show()
