import numpy as np
from lwlr import LWLR, GaussianKernel
import autograd as ad

class TrTimeAttackOnX:
    def __init__(self, X, Y, r, learner):
        self.init_X, self.init_Y = X, Y  # initial data
        self.r = r  # limit on how much the attacker can change
        self.learner = learner  # the learner to attack
        self.delta = np.random.rand(X.shape[0], X.shape[1])

    def loss(self, x_target, y_target, delta):
        theta_x = self.learner.forward(x_target, self.init_X+delta, self.init_Y+self.delta)
        return (1.0/2)*(theta_x-y_target)**2

    def fit(self, x_target, y_target, lr, epochs):
        # trains the model for self.epochs number of epochs on the entire training set

        for epoch in range(epochs):
            grad_fn = ad.grad(self.loss, 2)
            grad = grad_fn(x_target, y_target, self.delta)
            pred = self.learner.forward(x_target, self.init_X+self.delta, self.init_Y)
            print('Epoch %d/%d, Target y: %.4f, Current prediction: %.4f' % (epoch, epochs, y_target, pred))

            if np.linalg.norm(self.delta - grad * lr) > r:
                break
            self.delta = self.delta - grad * lr


if __name__ == '__main__':
    X = np.array([1., 1., 2., 4., 4., 5., 10., 10., 11.])
    Y = np.array([1., 2., 1., 6., 7., 7., 3., 4., 3.5])
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)

    r = 5

    x_target, y_target = 2., 10.

    model = LWLR(d=1, kernel=GaussianKernel, bandwidth=2, lbda=0)
    adversary = TrTimeAttackOnX(X, Y, 2, model)
    adversary.fit(x_target, y_target, lr=0.1, epochs=100)
    print("delta: ", adversary.delta)
    print("target y: ", y_target)
    print("model prediction at x: ", model.forward(x_target, X+adversary.delta, Y))

