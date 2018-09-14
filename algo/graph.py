import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

# load y-axis values from graphy.txt
y = np.loadtxt('graphy.txt', dtype=int)
# load x-axis values from graphx.txt
x = np.loadtxt('graphx.txt', dtype=int)

# convert epoch values to seconds
temp = x[0]
for i in range(len(x)):
    # print(i)
    x[i] -= temp

# the number (eg. 300) represents number of points to make between x.min() and x.max()
xnew = np.linspace(x.min(), x.max(), 50)
ynew = spline(x, y, xnew)

# plot graph
plt.ylabel('Number of interested audience')
plt.xlabel('Time (sec)')
plt.plot(xnew, ynew)
plt.savefig('../graph.png')
plt.show()