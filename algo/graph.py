import matplotlib.pyplot as plt
import numpy as np

# load y-axis values from graphy.txt
y = np.loadtxt('graphy.txt', dtype=int)
# load x-axis values from graphx.txt
x = np.loadtxt('graphx.txt', dtype=int)

# convert epoch values to seconds
temp = x[0]
for i in range(len(x)):
    print(i)
    x[i] -= temp

# plot graph
plt.plot(x, y, marker='x')
plt.show()