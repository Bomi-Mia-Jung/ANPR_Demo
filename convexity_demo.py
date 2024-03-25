import sys
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# import personal modules
from lwlr import LWLR, EpanechnikovKernel, TricubeKernel, LaplacianKernel, GaussianKernel, UniformKernel, AnovaRBFKernel, TriangularKernel
from attack import RepulsiveTrTimeAttack


def plot_x_convexity(adversary, lower_lim, upper_lim):
    # Make data.
    X = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.zeros(Y.shape)
    print(Y)

    Z = np.zeros(shape=X.shape)

    all_loss_points = []
    for i in range(X.shape[0]):
            loss_points, loss = adversary.loss(X[i], Y[i])
            all_loss_points.append(-1*loss_points)
            Z[i] = -1 * loss

    all_loss_points = np.array(all_loss_points)
    for i in range(all_loss_points.shape[1]):
        plt.plot(X, all_loss_points[:, i].flatten())

    plt.plot(X, Z)
    plt.show()


def plot_x_and_y_convexity(adversary, lower_lim, upper_lim):
    # Make data.
    X = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.arange(lower_lim, upper_lim, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(shape=X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            _, loss = adversary.loss(X[i, j], Y[i, j])
            # print(loss)
            Z[i, j] = -1*loss

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('x_delta')
    ax.set_ylabel('y_delta')
    ax.set_zlabel('model loss')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == '__main__':
    X = np.array([1., 1., 2., 4., 4., 5., 10., 10., 11.])
    Y = np.array([1., 2., 1., 6., 7., 7., 3., 4., 3.5])
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)

    r = 6.

    model = LWLR(d=1, kernel=GaussianKernel, bandwidth=2, lbda=0.1)
    adversary = RepulsiveTrTimeAttack(X, Y, r, model, lr=0.1, epochs=100)

    plot_x_convexity(adversary, -30, 30)
    plot_x_and_y_convexity(adversary, -30, 30)