import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


class Preprocessing:
    def __init__(self):
        data_path = 'datasets/the_trang_kmeans.csv'
        self.data = pd.read_csv(data_path)
        self.data_np = self.data.values

    def view(self):
        print(self.data.shape)

    def plot_raw(self):
        plt.scatter(self.data_np[:, 0], self.data_np[:, 1])
        plt.show()

    def normalization(self):
        self.data_np = np.array(self.data_np, dtype=np.float32)

        normalize_info = []
        for d in range(self.data_np.shape[1]):
            min_d = np.min(self.data_np[:, d])
            max_d = np.max(self.data_np[:, d])
            normalize_info.append([min_d, max_d])
            self.data_np[:, d] = (self.data_np[:, d] - min_d) / (max_d - min_d)

        np.save('models/normalize_info', np.array(normalize_info))

    def get_train_data(self):
        self.normalization()

        return self.data_np

    @staticmethod
    def draw_X_centers(data, centers):
        plt.scatter(data[:, 0], data[:, 1])
        plt.plot(centers[:, 0], centers[:, 1], 'rx')
        plt.show()

    def draw_X_centers_y(self, data, centers, y):
        plt.scatter(data[:, 0], data[:, 1], c=y)
        plt.plot(centers[:, 0], centers[:, 1], 'rx')
        plt.show()


if __name__ == '__main__':
    pp = Preprocessing()
    pp.view()
    pp.normalization()
    print(pp.data_np.dtype)
    pp.plot_raw()


