import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

# 生成一组随机数据
data = np.random.rand(200, 200)

# 对数据进行高斯滤波
data_smooth = filters.gaussian_filter(data, sigma=1)

# 计算梯度场
grad_x = filters.sobel(data_smooth, axis=1)
grad_y = filters.sobel(data_smooth, axis=0)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_dir = np.arctan2(grad_y, grad_x)

# 寻找局部极大值点
local_max = filters.maximum_filter(grad_mag, size=3)
local_max[grad_mag < 0.5*np.max(grad_mag)] = 0
local_max[local_max != grad_mag] = 0
print(local_max)
# 从局部极大值点出发追踪脊线
ridges = []
for i, j in zip(*np.where(local_max > 0)):
    ridge = [(i, j)]
    di, dj = np.round(np.cos(grad_dir[i, j])), np.round(np.sin(grad_dir[i, j]))
    while True:
        i, j = int(i + di), int(j + dj)
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1]):
            break
        if grad_mag[i, j] < 0.1*np.max(grad_mag):
            break
        ridge.append((i, j))
    if len(ridge) > 10:
        ridges.append(ridge)


# 将脊线可视化
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(data, cmap='gray', origin='lower')
for ridge in ridges:
    ax.plot([p[1] for p in ridge], [p[0] for p in ridge], 'r-', lw=2)
ax.axis('off')
plt.show()