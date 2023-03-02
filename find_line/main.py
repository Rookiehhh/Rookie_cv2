import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
from 边缘检测.Sobel import read_data, draw_contourf

filepath = os.path.abspath(f"../边缘检测/pd.txt")


def thr_data(filepath, savedir):
    data = read_data(filepath)
    for k, v in data.items():
        if k in ('X', 'Y'):
            continue
        # 获取x、y 坐标网格
        xx, yy = data['X'], data['Y']
        # 使用sobel梯度算子突出边缘
        grid = ndimage.sobel(v)
        grid = grid[1:-1, 1:-1]
        xx = xx[1:-1, 1:-1]
        yy = yy[1:-1, 1:-1]
        save_dirname = os.path.splitext(os.path.basename(filepath))[0]
        save_dir_path = os.path.join(savedir, save_dirname)
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
        for b in np.arange(0, 1, 0.05):
            thr_grid = grid.copy()
            thr = b * np.max(grid)
            thr_grid[thr_grid < thr] = 0
            # 查看当前梯度化效果
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            draw_contourf(fig, ax[0], xx, yy, grid, "Sobel", title=f'{k} Grid')
            draw_contourf(fig, ax[1], xx, yy, thr_grid, f"Sobel {b*100}%", title=f'{k} Grid')
            _save = os.path.join(save_dir_path, k)
            if not os.path.exists(_save):
                os.mkdir(_save)
            plt.savefig(os.path.join(_save, f"{b*100}%.png"))
            plt.close()


def eliminate_gjb(filepath):
    """
    剔除弓激波位形
    :param filepath:
    :return:
    """
    data = read_data(filepath)
    xx, yy = data['X'], data['Y']
    #  使用sobel卷积核计算梯度场, 强化边缘特征
    grad_x = ndimage.sobel(data['U_total'], axis=1)
    grad_y = ndimage.sobel(data['U_total'], axis=0)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # 裁剪边缘（存在异常）
    dw = 1
    grad_mag, xx, yy = map(lambda array: array[dw:-dw, dw:-dw], [grad_mag, xx, yy])
    scatter = find_last_max(grad_mag)   # 从右侧查找弓激波轮廓线位置, scatter 是检测点所在下标位置
    # points = list(zip(scatter[0], scatter[1]))
    # 剔除弓激波位形
    for y_index, x_index in zip(scatter[0], scatter[1]):
        grad_mag[y_index][x_index-3:x_index+3] = 0
    fig, ax = plt.subplots()
    draw_contourf(fig, ax, xx, yy, data['U_total'][dw:-dw, dw:-dw], "Sobel", title=f'ee')
    # 初步查找磁层顶位形位置, new_scatter 是检测点所在下标位置
    # new_scatter = find_last_max(grad_mag)
    # 以 y = 0 为界, 拆分为上下两部分进行剔除异常点操作
    xx0, yy0 = find_magnetopause(xx, yy, grad_mag)
    critical_rows = int(xx.shape[0] / 2)
    xx1, yy1 = find_magnetopause(xx[:critical_rows],
                                 np.flip(-yy[:critical_rows], axis=0),
                                 np.flip(grad_mag[:critical_rows], axis=0))
    xx2, yy2 = find_magnetopause(xx[critical_rows:], yy[critical_rows:], grad_mag[critical_rows:])
    sc_xx = np.concatenate((xx0, xx1, xx2), axis=0)
    sc_yy = np.concatenate((yy0, -yy1, yy2), axis=0)
    # # 剔除部分跳变点
    # new_scatter = remove_adjacent_mutation(new_scatter, 5)
    # # 以 y = 0 为界, 拆分为上下两部分进一步进行剔除异常点操作
    #
    # theta = np.arctan2(yy[new_scatter], xx[new_scatter])
    # start, end = find_max_increasing_subarray(theta)
    # new_scatter = (new_scatter[0][start:end+1], new_scatter[1][start:end+1])
    # # 剔除孤立点
    # new_scatter = remove_isolated_points(new_scatter[0], new_scatter[1], 2.0)
    print('&'*100)
    # print(xx[new_scatter][0], yy[new_scatter][0], np.sign(yy[new_scatter][0]))
    # print(xx[new_scatter][-1], yy[new_scatter][-1], np.sign(yy[new_scatter][-1]))
    print('&'*100)
    ax.scatter(xx[scatter], yy[scatter], c='r')
    ax.scatter(sc_xx, sc_yy, c='y')
    sc_y1, sc_x1 = smooth_xy(yy[scatter], xx[scatter])
    sc_y2, sc_x2 = smooth_xy(sc_yy, sc_xx)
    ax.plot(sc_x1, sc_y1, c='r')
    ax.plot(sc_x2, sc_y2, c='y')
    plt.show()


def find_magnetopause(xx, yy, grad_mag):
    # xx = xx.reshape(xx.shape)
    # yy = yy.reshape(yy.shape)
    # grad_mag = grad_mag.reshape(grad_mag.shape)
    scatter = find_last_max(grad_mag)
    # 剔除部分跳变点
    scatter = remove_adjacent_mutation(scatter, 5)
    theta = np.arctan2(yy[scatter], xx[scatter])
    start, end = find_max_increasing_subarray(theta)
    scatter = scatter[0][start:end+1], scatter[1][start:end+1]
    # 剔除孤立点
    scatter = remove_isolated_points(scatter[0], scatter[1], 2.0)
    # print('>>>')
    # print(xx[scatter], yy[scatter])
    # print('<<<')
    return xx[scatter], yy[scatter]


def find_last_max(grid):
    tmp_grid = grid.copy()
    for line in tmp_grid:
        line[line < np.nanmax(line) * 0.5] = np.nan
        unlast_index = np.where(np.logical_not(np.isnan(line)))[0][:-1]
        line[unlast_index] = np.nan
    # 筛选出最左侧较大梯度值的下标(即弓激波的位置)
    scatter = np.where(np.logical_not(np.isnan(tmp_grid)))

    return scatter


def smooth_xy(lx, ly):
    """数据平滑处理
    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    # 拟合函数方程
    f = lambda x, a, b, c, d, e: a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    x = np.array(lx)
    y = np.array(ly)
    # 通过拟合得到对应的方程各系数
    try:
        a, b, c, d, e = curve_fit(f, x, y)[0]
    except:
        return lx, ly
    # 根据拟合函数获取密集的曲线离散点数据
    new_x = np.linspace(np.nanmin(x), np.nanmax(x), 500)
    new_y = f(new_x, a, b, c, d, e)
    return new_x, new_y


def find_max_increasing_subarray(arr):
    """
    找到最大连续递增子数组，并返回其起始和结束下标。

    Parameters:
        arr (List[int]): 输入数组。

    Returns:
        Tuple[int, int]: 最大连续递增子数组的起始和结束下标。

    """
    max_start = 0          # 历史最大递增子数组的起始下标
    max_end = 0            # 历史最大递增子数组的结束下标
    current_start = 0      # 当前递增子数组的起始下标
    current_end = 0        # 当前递增子数组的结束下标

    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:    # 如果当前元素比上一个元素大，说明递增
            current_end = i        # 更新当前递增子数组的结束下标
            if current_end - current_start > max_end - max_start:
                # 如果当前递增子数组的长度比历史最大递增子数组的长度大，更新历史最大递增子数组的起始和结束下标
                max_start = current_start
                max_end = current_end
        else:
            current_start = i      # 如果当前元素不再递增，说明当前递增子数组结束，将起始和结束下标都设置为当前位置
            current_end = i

    return max_start, max_end    # 返回历史最大递增子数组的起始和结束下标作为结果


def remove_isolated_points(x, y, r=1.0):
    """
    剔除一组坐标点中的孤立点（半径1内没有其他坐标点）

    Parameters:
        x: numpy.ndarray, shape (n,)
            一组横坐标点
        y: numpy.ndarray, shape (n,)
            一组纵坐标点
        r: num
            孤立点判定半径
    Returns:
        filtered_x: numpy.ndarray, shape (m,)
            剔除孤立点后的横坐标点集
        filtered_y: numpy.ndarray, shape (n,)
            剔除孤立点后的纵坐标点集
    """
    points = np.column_stack((x, y))    # 将x和y组合成一个二维坐标点数组
    distances = np.sqrt(((points[:, np.newaxis, :] - points) ** 2).sum(axis=2)) # 计算每个点与其他点之间的距离
    neighborhood = distances <= r   # 计算每个点半径为r的邻域
    num_neighbors = neighborhood.sum(axis=1) - 1     # 计算每个点的邻居数量
    mask = num_neighbors > 0    # 保留至少有一个邻居的点
    filtered_points = points[mask]
    return filtered_points[:, 0], filtered_points[:, 1]


def remove_adjacent_mutation(scatter, thr):
    """
    剔除其中x坐标突变超过thr的坐标点
    Parameters
    ----------
    scatter: (numpy.ndarray, numpy.ndarray), 包含两个ndarry对象的元组,
     其中第一个ndarray 对象中是y坐标的下标, 第二是x坐标的下标
    thr: int
        突变阈值
    Returns
    -------
        剔除突变点后的坐标点下标信息集, 结构和scatter一致
    """
    # 初始化突变下标相对参考x轴坐标点
    new_x_index = scatter[1][0]
    for i, (y_index, x_index) in enumerate(zip(scatter[0], scatter[1])):
        # 标记所有超出阈值的点, 将其设置为-1
        if np.abs(new_x_index - x_index) > thr:
            scatter[0][i], scatter[1][i] = -1, -1
        else:
            new_x_index = x_index
    # 剔除所有的超出阈值点
    y_index = scatter[0][scatter[0] >= 0]
    x_index = scatter[1][scatter[1] >= 0]
    return y_index, x_index


if __name__ == '__main__':
    file_dir = os.path.abspath(f"../边缘检测/pddata")
    filenames = os.listdir(file_dir)
    filepaths = [os.path.join(file_dir, filename) for filename in filenames if filename[-3:] == 'txt']
    print(filepaths)
    for filepath in filepaths:
        # thr_data(filepath, file_dir)
        eliminate_gjb(filepath)

