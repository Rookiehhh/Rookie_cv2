import cv2
import numpy as np

# 旋转角度
alpha = np.deg2rad(30)

# 加载图片
img = cv2.imread(f"Circle.png")

# 获取图片尺寸
h, w, _ = img.shape

# 旋转变换仿射矩阵
M = np.array([
    [np.cos(alpha), -np.sin(alpha), 0],
    [np.sin(alpha), np.cos(alpha), 0],
])

#
# 执行仿射变换
dst = cv2.warpAffine(img, M, (w*2, h*2))

# 设置旋转中心
x0, y0 = w/2, h/2
M01 = np.array(
    [
        [1, 0, x0],
        [0, 1, x0],
        [0, 0, 1],
    ]

)
M02 = np.array([
    [1, 0, -x0],
    [0, 1, -y0],
    [0, 0, 1],
])
M_ = np.array([
    [np.cos(alpha), -np.sin(alpha), 0],
    [np.sin(alpha), np.cos(alpha), 0],
    [0, 0, 1]
])
_M = np.dot(M02, np.dot(M_, M01))
print(_M)
dst2 = cv2.warpAffine(img, _M[:2], (w*2, h*2))
# 显示原始图像和变换后的图像
cv2.imshow("RAW", img)
cv2.imshow("CAN", dst)
cv2.imshow("__", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()