import sys
import autograd.numpy as np
import math

# import personal modules
from lwlr import LWLR, EpanechnikovKernel, TricubeKernel, LaplacianKernel, GaussianKernel, UniformKernel, AnovaRBFKernel, TriangularKernel
from draggable import DraggablePlotTr, DraggablePlotTe
from attack import AttractiveTrTimeAttack
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    Lasso,
    Ridge
)


# side note: if you're going to write your own regressor to plug in:
# regressor.fit(X, Y, W) should return fitted regressor
# X = (n, d), Y = (n, ), W = (d, )
# regressor.predict(x_input) should return a scalar y value


if __name__ == '__main__':

    # plt.ion()

    d = 1  # dimensions of X
    r = 2.  # radius of circle within which you can move each data point

    x_range = (0, 30)
    y_range = (0, 20)

    # training data
    X = [1., 1., 2., 4., 4., 5., 10., 10., 11.]
    Y = [1., 2., 1., 6., 7., 7., 3., 4., 3.5]

    # X = [1., 2., 3.]
    # Y = [1., 1., 1.]

    test_x = [3, 10.5, 25]  # test the predicted outputs at these three points (like in class)

    data = tuple(zip(X, Y))

    bandwidth = 2
    lbda = 0.1

    kernel = GaussianKernel  # kernel function to use for weights
    regressors = {'OLS': LinearRegression(), 'Huber': HuberRegressor(), 'Lasso': Lasso(), 'Ridge': Ridge()}  # , 'RANSAC': RANSACRegressor()}
    models = {'{}'.format(name): LWLR(d, kernel, regressor, bandwidth, lbda) for (name, regressor) in regressors.items()}  # initialize locally weighted lin reg model

    tr_plot = DraggablePlotTr(points=data, test_points=test_x, r=r, domain=x_range, range=y_range, title="ANPR Draggable Tr Set", models=models)

    X = np.array(X, dtype=np.float64)
    # X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.array(Y, dtype=np.float64)
    # Y = np.reshape(Y, (Y.size, 1))  # (n, 1)
    model = LWLR(d, kernel, None, bandwidth, lbda)
    adversary = AttractiveTrTimeAttack(X, Y, r, model, lr=0.1, epochs=100)

    te_plot = DraggablePlotTe(points=data, test_points=[25.], r=r, domain=x_range, range=y_range, title="ANPR Draggable Target", model=model, attack=adversary)
