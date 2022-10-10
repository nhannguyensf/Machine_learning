import numpy as np
from preprocessing import Preprocessing


class LinearRegression:
    def __init__(self):
        self.pp = Preprocessing()

    def train(self):
        data = self.pp.get_train_data()
        X = data[:, 0]
        y = data[:, 1]
        X_ones = np.concatenate([np.ones((X.shape[0], 1)), X.reshape(-1, 1)], axis=1)
        y = y.reshape((-1, 1))

        phase1 = X_ones.T.dot(X_ones)
        phase2 = X_ones.T.dot(y)

        w = np.linalg.pinv(phase1).dot(phase2)
        w = w.reshape(-1, )

        x_draw = np.array([0, 1])
        y_draw = w[0] + x_draw * w[1]

        self.pp.plot_line(X=data, x_draw=x_draw, y_draw=y_draw)


if __name__ == '__main__':
    linreg = LinearRegression()
    linreg.train()
