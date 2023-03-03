"""
点集是指坐标点的集。已知二维笛卡尔坐标系中有很多坐标点, 需要找到包围这些坐标点的最小外包四边形或者圆,
这里的最小指的是最小面积
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
mpl.rcParams['font.family'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 创建点集
# points = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5], [6, 7], [2, 11]], np.int32)
points = np.array([[1, 1], [3, 2], [5, 7], [10, 5], [10, 7], [3, 3], [5, 0], [10, 10]], np.int32)
# 最小外包旋转矩形
rotateRect = cv2.minAreaRect(points)
# 最小外包圆
circle = cv2.minEnclosingCircle(points)
# 最小外包直立矩形
rect = cv2.boundingRect(points)
# 最小凸包
convexhull = cv2.convexHull(points)
# 最小外包三角形
_, triangle = cv2.minEnclosingTriangle(points)
print(rotateRect)
print(circle)
print(rect)
print(convexhull)
print(triangle)
fig, ax = plt.subplots()
plt.scatter(*points.T)
x0, y0 = rotateRect[0]
width, height = rotateRect[1]
angle = rotateRect[2]
rectangle = patches.Rectangle(xy=(x0-width/2, y0-height/2), width=width, height=height, angle=angle,
                              ec='r', fc='none', rotation_point='center', label='最小旋转外包矩形')
ax.add_patch(rectangle)
ax.add_patch(patches.Circle(xy=circle[0], radius=circle[1], facecolor='none', ec='b', label='最小外包圆'))
# 直立矩形左下顶点
rect_x0, rect_y0 = rect[0], rect[1]
# 直立矩形高和宽
rect_h = rect[3] - rect_y0
rect_w = rect[2] - rect_x0
ax.add_patch(patches.Rectangle(xy=(rect_x0, rect_y0), width=rect_w, height=rect_h, facecolor='none', ec='c', label='最小外包直立矩形'))
# 最小凸包
ax.fill(*convexhull.reshape(-1, 2).T, ec='g', fc='none', label='最小凸包')
# 最小外包三角形
ax.fill(*triangle.reshape(-1, 2).T, ec='y', fc='none', label='最小外包三角形')
plt.xlim(-10, 20)
plt.ylim(-10, 20)
ax.set_aspect(1)
fig.legend()
plt.show()
