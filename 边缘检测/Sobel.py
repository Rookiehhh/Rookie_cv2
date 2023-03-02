"""
边缘是像素值发生跃迁的位置, 是图像的显著特征之一, 在图像特征提取, 对象识别, 模式识别中都有重要作用
索贝尔sobel算子
    水平方向

        Gx =
"""
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from functools import wraps


def read_data(filepath):
    pf = pd.read_csv(filepath)
    # pf.pop('Unnamed: 0')
    ret = dict()
    for k, v in pf.items():
        ret[k] = v.values.reshape((65, 65))
    return ret


def draw_contourf(fig, ax, X, Y, data, name, title):
    ax.set_title(title)
    ax.set_aspect(1)
    levels = np.linspace(np.min(data), np.max(data), 50)
    # ax.contour(X, Y, data, levels=levels)
    cf = ax.contourf(X, Y, data, levels=levels)

    cbar = fig.colorbar(cf, ax=ax, )
    cbar.ax.set_title(name)


def grad(func):
    @wraps(func)
    def inner(data):
        grad_x, grad_y = func(data)
        grad_x = signal.convolve2d(data, grad_x, mode='same')
        grad_y = signal.convolve2d(data, grad_y, mode='same')
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad_mag
    return inner


def grad_abs_max(func):
    @wraps(func)
    def inner(data):
        ret = func(data)
        max_grad = np.zeros_like(data)
        for k_i in ret:
            grid_i = np.fabs(signal.convolve2d(data, k_i, mode='same'))
            # print(grid_i.shape, max_grad.shape)
            max_grad = np.maximum(max_grad, grid_i)
        return max_grad
    return inner


@grad
def sobel(data):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    return sobel_x, sobel_y


@grad
def scharr(data):
    scharr_x = np.array([[3, 0, -3],
                         [10, 0, 10],
                         [3, 0, -3]])
    scharr_y = np.array([[3, 10, 3],
                         [0, 0, 0],
                         [-3, -10, -3]])
    return scharr_x, scharr_y


@grad_abs_max
def kirsch(data):
    k1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    k2 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    k3 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    k4 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    k5 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    k6 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    k7 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    k8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    return k1, k2, k3, k4, k5, k6, k7, k8


@grad_abs_max
def robinson(data):
    r1 = np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
    r2 = np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]])
    r3 = np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]])
    r4 = np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]])
    r5 = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
    r6 = np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]])
    r7 = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])
    r8 = np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
    return r1, r2, r3, r4, r5, r6, r7, r8


def laplacian(data):
    """拉普拉斯算子"""
    l0 = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
    return signal.convolve2d(data, l0, mode='same')


def run():
    dir_name = 'png'
    dir_path = os.path.abspath(dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    data = read_data(f'pd.txt')
    x, y = data['X'], data["Y"]
    dw = 1
    xx, yy = x[dw:-dw, dw:-dw], y[dw:-dw, dw:-dw]
    vars = data.keys()
    for var in vars:
        fig, ax = plt.subplots(3, 2, figsize=(30, 20))
        ax = ax.reshape((-1, ))
        raw_func = lambda x: x
        raw_func.__name__ = str(var) + ' RAW'
        funcs = [raw_func,
                sobel, scharr, kirsch, robinson, laplacian]
        fig.suptitle(var, size=20)
        for i, func in enumerate(funcs):
            img = func(data[var])[dw:-dw, dw:-dw]
            img[img < np.max(img)*0.45] = 0
            draw_contourf(fig, ax[i], xx, yy, img, func.__name__, title=func.__name__)
        plt.savefig(os.path.join(dir_path, f"{var}.png"))
        # plt.show()


if __name__ == '__main__':
    run()
