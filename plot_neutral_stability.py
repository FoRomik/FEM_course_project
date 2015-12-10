#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

c_10 = 1.0
c_20 = 1.0
k_mn = {(0,1): 3.8317 , (1,1): 1.8412, (3,3): 11.3459}

besselmode = (0,1)
slope = ( -c_20 * (c_20 - 2 * c_10) ) / k_mn[besselmode]**2.

k = np.linspace(0,1,100)
D = slope * k

fig = plt.figure(facecolor = 'w', edgecolor = 'w', figsize = (5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(k, D, linewidth = 2)
ax.set_xlabel('Reaction rate constant (k)', fontsize = 'large')
ax.set_ylabel('Diffusion constant (D)', fontsize = 'large')
plt.show()
