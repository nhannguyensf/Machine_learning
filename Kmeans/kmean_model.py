import numpy as np
from preprocessing import Preprocessing
from scipy.spatial.distance import cdist
import time


class KMean:
    def __init__(self):
        self.k = 3
        self.pp = Preprocessing()

    def train(self):
        data = self.pp.get_train_data()
        centers = data[np.random.choice(data.shape[0], self.k, replace=False)]
        # self.pp.draw_X_centers(data=data, centers=centers)

        while True:
            y = np.argmin(cdist(data, centers), axis=1)
            print(y.shape)

            self.pp.draw_X_centers_y(data, centers, y)

            new_centers = []
            for i in range(self.k):
                new_centers.append(np.mean(data[y == i], axis=0))
            new_centers = np.array(new_centers)

            print(centers)
            print(new_centers)

            time.sleep(0.5)

            centers_set = set([tuple(c) for c in centers])
            new_centers_set = set([tuple(c) for c in new_centers])

            if len(centers_set - new_centers_set) == 0:
                break

            centers = new_centers


if __name__ == '__main__':
    k_mean = KMean()
    k_mean.train()
