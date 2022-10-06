import numpy as np

x = np.array([142, 147, 152, 157, 162])
y = np.array([42, 47, 52, 57, 62])

r = np.corrcoef(x, y)
print(r)
