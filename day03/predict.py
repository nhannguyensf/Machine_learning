import numpy as np
from scipy.spatial.distance import cdist


class KMeanPredict:
    def __init__(self):
        self.centers = np.load('models/centers.npy')
        self.normalize_info = np.load('models/normalize_info.npy')
        self.classes_name = {0: 'Hơi thừa cân', 1: 'Hơi thiếu cân', 2: 'Thân hình chuẩn'}

        # print('centers:', self.centers)

    def normalization(self, x):
        x = np.array(x, dtype=np.float32)
        for d in range(x.shape[1]):
            x[:, d] = (x[:, d] - self.normalize_info[d][0]) / (self.normalize_info[d][1] - self.normalize_info[d][0])
        return x

    def predict(self, x_new):
        x_new = np.array([x_new])
        x_new = self.normalization(x_new)
        # print(x_new)
        y = np.argmin(cdist(x_new, self.centers), axis=1)
        print(self.classes_name[y[0]])


if __name__ == '__main__':
    _x_new = [175, 700]
    print('Chiều cao:', _x_new[0])
    print('Cân nặng:', _x_new[1])
    k_mean_pred = KMeanPredict()
    k_mean_pred.predict(x_new=_x_new)
