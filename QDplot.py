import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("outputnew.txt", delimiter=' ')
plt.plot(data[:-1, 0],data[:-1, 1])
plt.xlabel("N")
plt.ylabel("price")
plt.show()