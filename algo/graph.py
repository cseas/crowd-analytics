import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt('graphy.txt', dtype=int)

# load x-axis values from graphx.txt and convert epoch values to seconds
x = np.loadtxt('graphx.txt', dtype=int)
temp = x[0]
for i in range(len(x)):
    print(i)
    x[i] -= temp
print(x)
plt.plot(x, y, marker='x')
plt.show()