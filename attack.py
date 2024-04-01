import autograd.numpy as np
from lwlr import LWLR, GaussianKernel
import autograd as ad
import math
import copy
from sklearn.linear_model import LinearRegression


class AttractiveTrTimeAttack:
    def __init__(self, X, Y, r, learner, lr=0.1, epochs=100):
        self.init_X, self.init_Y = X, Y  # initial data
        self.r = r  # limit on how much the attacker can change
        self.learner = learner  # the learner to attack
        self.lr = lr  # learning rate
        self.epochs = epochs  # how many epochs to train for
        self.x_delta = np.random.rand(*[X.shape[i] for i in range(X.ndim)])
        self.y_delta = np.random.rand(*[Y.shape[i] for i in range(Y.ndim)])

    def loss(self, x_target, y_target, x_delta, y_delta):
        theta_x = self.learner.forward(x_target, copy.copy(self.init_X)+x_delta, copy.copy(self.init_Y)+y_delta)
        return (1.0/2)*(theta_x-y_target)**2

    def fit(self, x_target, y_target):
        # trains the model for self.epochs number of epochs on the entire training set
        x_target = copy.copy(x_target)
        y_target = copy.copy(y_target)

        self.x_delta = np.random.rand(*[self.x_delta.shape[i] for i in range(self.x_delta.ndim)])  # random start
        self.y_delta = np.random.rand(*[self.y_delta.shape[i] for i in range(self.y_delta.ndim)])
        # print(self.x_delta)  # check if random start is actually different every time

        b1 = 0.9
        b2 = 0.999
        eps = 10 ** -8
        m = np.zeros(self.x_delta.shape)
        v = np.zeros(self.x_delta.shape)

        for epoch in range(self.epochs):
            grad_fn = ad.grad(self.loss, [2, 3])
            grad = grad_fn(x_target, y_target, self.x_delta, self.y_delta)
            grad = np.array(grad)

            # adam
            m = (1 - b1) * grad + b1 * m  # first  moment estimate
            v = (1 - b2) * (grad ** 2) + b2 * v  # second moment estimate
            mhat = m / (1 - b1 ** (epoch + 1))  # bias correction?
            vhat = v / (1 - b2 ** (epoch + 1))
            delta = self.lr * mhat / (np.sqrt(vhat) + eps)
            new_x_delta = self.x_delta - delta[0]
            new_y_delta = self.y_delta - delta[1]
            diff = np.sqrt(np.square(new_x_delta)+np.square(new_y_delta))
            new_x_delta = np.where(diff <= self.r, new_x_delta, self.x_delta)
            new_y_delta = np.where(diff <= self.r, new_y_delta, self.y_delta)

            self.x_delta = new_x_delta
            self.y_delta = new_y_delta

            pred = self.learner.forward(x_target, self.init_X+self.x_delta, self.init_Y+self.y_delta)
            print('Epoch %d/%d, Target y: %.4f, Current prediction: %.4f' % (epoch, self.epochs, y_target, pred))

        return self.init_X + self.x_delta, self.init_Y + self.y_delta


class RepulsiveTrTimeAttack:
    def __init__(self, X, Y, r, learner, lr=0.1, epochs=100):
        self.init_X, self.init_Y = X, Y  # initial data
        # X = (n, d), Y = (n, 1)

        self.r = r  # limit on how much the attacker can change
        self.learner = learner  # the learner to attack
        self.lr = lr  # learning rate
        self.epochs = epochs  # how many epochs to train for
        self.x_delta = np.random.rand(X.shape[0], X.shape[1])
        self.y_delta = np.random.rand(Y.shape[0], Y.shape[1])

    def loss(self, x_delta, y_delta):
        curr_X = copy.copy(self.init_X)
        curr_X[0] = curr_X[0] + x_delta
        # curr_X = curr_X+x_delta

        # print(curr_X)
        curr_Y = copy.copy(self.init_Y)
        curr_Y[0] = curr_Y[0] + y_delta
        # curr_Y = curr_Y+y_delta

        # print(self.init_Y.shape)
        # print(self.init_Y[0, 0])
        # print(y_delta)
        total_loss = 0.0
        loss_points = []
        for i in range(self.init_X.size):
            output = self.learner.forward(self.init_X[i, :].item(), curr_X, curr_Y).item()
            loss = (output-self.init_Y[i, 0])**2
            loss_points.append(-1 * loss)
            total_loss -= loss
            # print(total_loss)
        return np.array(loss_points), total_loss/self.init_X.shape[0]

    def fit(self):
        # trains the model for self.epochs number of epochs on the entire training set

        self.x_delta = np.random.rand(self.x_delta.shape[0], self.x_delta.shape[1])  # random start
        self.y_delta = np.random.rand(self.y_delta.shape[0], self.y_delta.shape[1])
        # print(self.x_delta)  # check if random start is actually different every time

        b1 = 0.9
        b2 = 0.999
        eps = 10 ** -8
        m = np.zeros(self.x_delta.shape)
        v = np.zeros(self.x_delta.shape)

        for epoch in range(self.epochs):
            grad_fn = ad.grad(self.loss, [0, 1])
            grad = grad_fn(self.x_delta, self.y_delta)
            grad = np.array(grad)

            # adam
            m = (1 - b1) * grad + b1 * m  # first  moment estimate
            v = (1 - b2) * (grad ** 2) + b2 * v  # second moment estimate
            mhat = m / (1 - b1 ** (epoch + 1))  # bias correction?
            vhat = v / (1 - b2 ** (epoch + 1))
            delta = self.lr * mhat / (np.sqrt(vhat) + eps)
            new_x_delta = self.x_delta - delta[0]
            new_y_delta = self.y_delta - delta[1]
            diff = np.sqrt(np.square(new_x_delta)+np.square(new_y_delta))
            new_x_delta = np.where(diff <= self.r, new_x_delta, self.x_delta)
            new_y_delta = np.where(diff <= self.r, new_y_delta, self.y_delta)

            self.x_delta = new_x_delta
            self.y_delta = new_y_delta

            print('Epoch %d/%d, Current loss: %.4f' % (epoch, self.epochs, self.loss(self.x_delta, self.y_delta)))

        return self.init_X + self.x_delta, self.init_Y + self.y_delta


if __name__ == '__main__':
    X = np.array([1., 1., 2., 4., 4., 5., 10., 10., 11.])
    Y = np.array([1., 2., 1., 6., 7., 7., 3., 4., 3.5])
    X = np.reshape(X, (X.size, 1))  # (n, d)
    Y = np.reshape(Y, (Y.size, 1))  # (n, 1)

    r = 6.

    # attractive attack test
    #
    # x_target, y_target = 2., -10.
    #
    # model = LWLR(d=1, kernel=GaussianKernel, bandwidth=2, lbda=0.1)
    # adversary = AttractiveAttackOnX(X, Y, r, model, lr=0.1, epochs=100)
    # changed_X, changed_Y = adversary.fit(x_target, y_target)
    # print("X delta: ", adversary.x_delta)
    # print("Y delta: ", adversary.y_delta)
    # print("target y: ", y_target)
    # print("model prediction at x: ", model.forward(x_target, changed_X, changed_Y))

    #  repulsive attack test

    x_target = 5.

    model = LWLR(d=1, kernel=GaussianKernel, regressor=LinearRegression(), bandwidth=2, lbda=0.1)
    adversary = RepulsiveTrTimeAttack(X, Y, r, model, lr=0.1, epochs=100)
    changed_X, changed_Y = adversary.fit()
    print("X delta: ", adversary.x_delta)
    print("Y delta: ", adversary.y_delta)
    print("model original pred at x: ", model.forward(x_target, X, Y))
    print("model attacked pred at x: ", model.forward(x_target, changed_X, changed_Y))


