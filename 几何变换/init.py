import cv2
import numpy as np

# 加载图像
img = cv2.imread('Circle.png')

# 获取图像尺寸
rows, cols, _ = img.shape

# 定义变换前后的三个点的坐标
src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
dst_pts = np.float32([[cols*0.2, rows*0.2], [cols*0.8, rows*0.2], [cols*0.1, rows*0.9]])

# 计算仿射变换矩阵
M = cv2.getAffineTransform(src_pts, dst_pts)
print(M)
# 执行仿射变换
dst = cv2.warpAffine(img, M, (cols, rows))

# 显示原始图像和变换后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
