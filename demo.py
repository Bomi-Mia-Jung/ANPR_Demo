import sys
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.backend_bases import MouseEvent


# import personal modules
from lwlr import LWLR, GaussianKernel
from draggable import DraggablePlotTr, DraggablePlotTe
from attack import TrTimeAttackOnX


if __name__ == '__main__':

    # plt.ion()

    d = 1  # dimensions of X
    r = 2.  # radius of circle within which you can move each data point

    x_range = (0, 30)
    y_range = (0, 20)

    # training data
    X = [1., 1., 2., 4., 4., 5., 10., 10., 11.]
    Y = [1., 2., 1., 6., 7., 7., 3., 4., 3.5]

    test_x = [3, 10.5, 25]  # test the predicted outputs at these three points (like in class)

    data = tuple(zip(X, Y))

    kernel = GaussianKernel  # kernel function to use for weights
    model = LWLR(d, kernel, 2, lbda=0)  # initialize locally weighted lin reg model

    tr_plot = DraggablePlotTr(points=data, test_points=test_x, r=r, domain=x_range, range=y_range, title="ANPR Draggable Tr Set", model=model)

    X = np.array(X)
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.array(Y)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)
    adversary = TrTimeAttackOnX(X, Y, r, model, lr=0.05, epochs=100)

    te_plot = DraggablePlotTe(points=data, test_points=[25], r=r, domain=x_range, range=y_range, title="ANPR Draggable Target Points", model=model, attack=adversary)
