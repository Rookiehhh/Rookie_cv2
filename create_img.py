import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


circle = Circle((2, 3), 1, fc='k')
fig, ax = plt.subplots()
plt.xlim(0, 5)
plt.ylim(0, 5)
ax.add_patch(circle)
ax.set_aspect(1)
# ax.axis("off")
plt.savefig("几何变换/Circle.png")