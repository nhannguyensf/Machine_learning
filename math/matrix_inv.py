import numpy as np

# 5x0 + 4x1 + x2 = 1
# 6x0 + 3x1 + x2 = 3
# 10x0 + 6x1 + 2x2 = 7

# phép chuyển vị có kết quả là gì?

A = np.array([[5, 4, 1],
              [6, 3, 1],
              [10, 6, 2]])

b = np.array([[1],
              [3],
              [7]])

x = np.linalg.pinv(A).dot(b)

print(x)
