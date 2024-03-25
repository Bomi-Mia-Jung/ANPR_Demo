import sys
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# import personal modules
from lwlr import LWLR, EpanechnikovKernel, TricubeKernel, LaplacianKernel, GaussianKernel, UniformKernel, AnovaRBFKernel, TriangularKernel
from attack import RepulsiveAttackOnX


if __name__ == '__main__':
    X = np.array([1., 1., 2., 4., 4., 5., 10., 10., 11.])
    Y = np.array([1., 2., 1., 6., 7., 7., 3., 4., 3.5])
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)

    r = 6.

    model = LWLR(d=1, kernel=GaussianKernel, bandwidth=2, lbda=0.1)
    adversary = RepulsiveAttackOnX(X, Y, r, model, lr=0.1, epochs=100)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-10, 10, 0.25)
    # X = X.reshape((1, X.size))
    Y = np.arange(-10, 10, 0.25)
    # Y = Y.reshape((1, Y.size))
    # Y = np.zeros(Y.shape)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros(shape=X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = (-1 * adversary.loss(X[i, j], Y[i, j]))

    # Plot the surface.
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
