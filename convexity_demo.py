import sys
import numpy as np
import math
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from labellines import *
from mpl_axes_aligner import align

# import personal modules
from lwlr import LWLR, EpanechnikovKernel, TricubeKernel, LaplacianKernel, GaussianKernel, UniformKernel, AnovaRBFKernel, TriangularKernel
from attack import RepulsiveTrTimeAttack


def plot_x_convexity(adversary, lower_lim, upper_lim, move_idx=0):
    # Make data.
    X = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.zeros(Y.shape)
    # print(Y)

    Z = np.zeros(shape=X.shape)

    all_loss_points = []
    for i in range(X.shape[0]):
            loss_points, loss = adversary.loss(X[i], Y[i], change_all=False, move_idx=move_idx)
            all_loss_points.append(-1*loss_points)
            Z[i] = -1 * loss

    fig, ax1 = plt.subplots()
    ax1.set_title('point {} + x delta vs. defender loss'.format(move_idx))
    ax1.set_xlabel('x delta')
    ax1.set_ylabel('training loss (defender)')

    all_loss_points = np.array(all_loss_points)
    for i in range(all_loss_points.shape[1]):
        ax1.plot(X, all_loss_points[:, i].flatten(), label='point {}'.format(i))

    ax1.plot(X.flatten(), Z.flatten())

    ax2 = plt.twinx(ax1)
    ax2.set_ylabel('original data (y)')
    ax2.plot(copy.copy(adversary.init_X)-adversary.init_X[move_idx, :], adversary.init_Y, 'bo', markersize=4)
    ax2.plot([0], adversary.init_Y[move_idx], 'ro', markersize=4)

    # plt.legend(loc='best')
    labelLines(ax1.get_lines(), zorder=2.5)
    plt.show()


def plot_y_convexity(adversary, lower_lim, upper_lim, move_idx=0):
    # Make data.
    X = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.arange(lower_lim, upper_lim, 0.25)
    X = np.zeros(X.shape)
    # print(Y)

    Z = np.zeros(shape=Y.shape)

    all_loss_points = []
    for i in range(Y.shape[0]):
            loss_points, loss = adversary.loss(X[i], Y[i], change_all=False, move_idx=move_idx)
            all_loss_points.append(-1*loss_points)
            Z[i] = -1 * loss

    fig, ax1 = plt.subplots()
    ax1.set_title('point {} + y delta vs. defender loss'.format(move_idx))
    ax1.set_xlabel('y delta')
    ax1.set_ylabel('training loss (defender)')

    all_loss_points = np.array(all_loss_points)
    for i in range(all_loss_points.shape[1]):
        ax1.plot(Y, all_loss_points[:, i].flatten(), label='point {}'.format(i))

    ax1.plot(Y.flatten(), Z.flatten())

    ax2 = plt.twinx(ax1)
    ax2.set_ylabel('original data (x)')
    ax2.plot(copy.copy(adversary.init_Y)-adversary.init_Y[move_idx, :], adversary.init_X, 'bo', markersize=4)
    ax2.plot([0], adversary.init_X[move_idx], 'ro', markersize=4)

    # plt.legend(loc='best')
    labelLines(ax1.get_lines(), zorder=2.5)
    plt.show()



def plot_x_and_y_convexity(adversary, lower_lim, upper_lim, change_all=True, move_idx=0):
    # Make data.
    X = np.arange(lower_lim, upper_lim, 0.25)
    Y = np.arange(lower_lim, upper_lim, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(shape=X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            _, loss = adversary.loss(X[i, j], Y[i, j], change_all=change_all, move_idx=move_idx)
            # print(loss)
            Z[i, j] = -1*loss

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if change_all:
        ax.set_title('x, y delta vs. defender loss')
    else:
        ax.set_title('point {} + x, y delta vs. defender loss'.format(move_idx))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('x delta')
    ax.set_ylabel('y delta')
    ax.set_zlabel('training loss (defender)')

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

    plot_x_convexity(adversary, -30, 30, move_idx=5)
    plot_y_convexity(adversary, -30, 30, move_idx=5)
    plot_x_and_y_convexity(adversary, -30, 30, change_all=False, move_idx=5)