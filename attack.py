import numpy as np
from lwlr import LWLR, GaussianKernel
import autograd as ad
from autograd.misc.optimizers import adam
import math

class TrTimeAttackOnX:
    def __init__(self, X, Y, r, learner, lr=0.1, epochs=100):
        self.init_X, self.init_Y = X, Y  # initial data
        self.r = r  # limit on how much the attacker can change
        self.learner = learner  # the learner to attack
        self.lr = lr  # learning rate
        self.epochs = epochs  # how many epochs to train for
        self.x_delta = np.random.rand(X.shape[0], X.shape[1])
        self.y_delta = np.random.rand(Y.shape[0], Y.shape[1])

    def loss(self, x_target, y_target, x_delta, y_delta):
        theta_x = self.learner.forward(x_target, self.init_X+x_delta, self.init_Y+y_delta)
        return (1.0/2)*(theta_x-y_target)**2

    def fit(self, x_target, y_target):
        # trains the model for self.epochs number of epochs on the entire training set

        self.x_delta = np.random.rand(self.x_delta.shape[0], self.x_delta.shape[1])  # random start
        self.y_delta = np.random.rand(self.y_delta.shape[0], self.y_delta.shape[1])
        # print(self.x_delta)  # check if random start is actually different every time

        for epoch in range(self.epochs):
            grad_fn = ad.grad(self.loss, [2, 3])
            grad = grad_fn(x_target, y_target, self.x_delta, self.y_delta)
            pred = self.learner.forward(x_target, self.init_X+self.x_delta, self.init_Y+self.y_delta)
            print('Epoch %d/%d, Target y: %.4f, Current prediction: %.4f' % (epoch, self.epochs, y_target, pred))

            new_x_delta = self.x_delta - grad[0] * self.lr
            new_x_delta = np.where(np.abs(new_x_delta) <= self.r, new_x_delta, self.x_delta)
            new_y_delta = self.y_delta - grad[1] * self.lr
            new_y_delta = np.where(np.abs(new_y_delta) <= self.r, new_y_delta, self.y_delta)

            self.x_delta = new_x_delta
            self.y_delta = new_y_delta

        return self.init_X + self.x_delta, self.init_Y + self.y_delta


if __name__ == '__main__':
    X = np.array([1., 1., 2., 4., 4., 5., 10., 10., 11.])
    Y = np.array([1., 2., 1., 6., 7., 7., 3., 4., 3.5])
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)

    r = 6.

    x_target, y_target = 2., 10.

    model = LWLR(d=1, kernel=GaussianKernel, bandwidth=2, lbda=0.1)
    adversary = TrTimeAttackOnX(X, Y, r, model, lr=0.1, epochs=100)
    changed_X, changed_Y = adversary.fit(x_target, y_target)
    print("X delta: ", adversary.x_delta)
    print("Y delta: ", adversary.y_delta)
    print("target y: ", y_target)
    print("model prediction at x: ", model.forward(x_target, changed_X, changed_Y))

