import matplotlib as plt
import numpy as np

y = np.loadtxt('graphy.txt', dtype=int)
x = np.loadtxt('graphx.txt', dtype=int)

plot(x, y)