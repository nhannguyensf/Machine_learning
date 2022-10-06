import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-1000, 1000 + 1, 1)
y = 5 * (x ** 4) - 12 * (x ** 2) + 2

plt.plot(x, y, 'g')
plt.show()
