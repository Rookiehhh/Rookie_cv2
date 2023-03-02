"""
点集是指坐标点的集。已知二维笛卡尔坐标系中有很多坐标点, 需要找到包围这些坐标点的最小外包四边形或者圆,
这里的最小指的是最小面积
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
# 创建点集
# points = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5], [6, 7], [2, 11]], np.int32)
points = np.array([[1, 1], [3, 2], [5, 7], [10, 5], [10, 7], [3, 3]], np.int32)
rotateRect = cv2.minAreaRect(points)
print(rotateRect)
# vertices = cv2.boxPoints(rotateRect)
# print(vertices)
tmp_points = points.T
fig, ax = plt.subplots()
plt.scatter(tmp_points[0], tmp_points[1])
x0, y0 = rotateRect[0]
width, height = rotateRect[1]
angle = rotateRect[2]
rectangle = patches.Rectangle(xy=(x0-width/2, y0-height/2), width=width, height=height, angle=angle,
                              ec='r', fc='none', rotation_point='center')
ax.add_patch(rectangle)
plt.xlim(0, 12)
plt.ylim(0, 12)
ax.set_aspect(1)
plt.show()
