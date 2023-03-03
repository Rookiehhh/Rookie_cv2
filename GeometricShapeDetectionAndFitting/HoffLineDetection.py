"""
霍夫直线检测
"""
import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def HTLine(image, stepTheta=1, setpRho=1):
    # 宽 高
    rows, cols = image.shape
    # 图像中可能出现的最大垂线的长度
    L = round(math.sqrt(pow(rows-1, 2.0)+pow(cols-1, 2.0))) + 1
    # 初始化投票器
    numtheta = int(180.0/stepTheta)
    numRho = int(2*L/setpRho + 1)   # 发生偏移
    accumulator = np.zeros((numRho, numtheta), np.int32)
    # 建立字典
    accuDict = {}
    for k1 in range(numRho):
        for k2 in range(numtheta):
            accuDict[(k1, k2)] = []
    # 投票计数
    for y in range(rows):
        for x in range(cols):
            if(image[y][x] == 255): # 只对边缘点做霍夫变换
                for m in range(numtheta):
                    # 对每个角度, 计算对应的 rho 值
                    rho = x*math.cos(stepTheta*m/180.0*math.pi) + y*math.sin(stepTheta*m/180.0*math.pi)
                    # 计算投票哪个区域
                    n = int(round(rho+L)/setpRho)
                    # 投票数+1
                    accumulator[n, m] += 1
                    # 记录该点
                    accuDict[(n, m)].append((y, x))

    return accumulator, accuDict


if __name__ == '__main__':
    I = cv2.imread(os.path.abspath(r"./southeast.jpg"))
    # 将图像转为灰度图
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny 边缘检测
    edge = cv2.Canny(blurred, 50, 200)

    # 显示二值化边缘
    cv2.imshow("edge", edge)
    # 霍夫直线检测
    accumulator, accuDict = HTLine(edge, 1, 1)
    # 计数器的二维直方图显示
    rows, cols = accumulator.shape
    fig= plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.mgrid[0:rows:1, 0:cols:1]
    surf = ax.plot_wireframe(X, Y, accumulator, cstride=1, rstride=1, color='gray')
    ax.set_xlabel(u"$\\rho$")
    ax.set_ylabel(u"$\\theta$")
    ax.set_zlabel("accumulator")
    ax.set_zlim3d(0, np.max(accumulator))
    # plt.show()
    # 计数器的灰度级显示
    grayAccu = accumulator/float(np.max(accumulator))
    grayAccu = 255*grayAccu
    grayAccu = grayAccu.astype(np.uint8)
    # 只画出投票数大于60的直线
    voteThresh = 300
    for r in range(rows):
        for c in range(cols):
            if accumulator[r][c] > voteThresh:
                points = accuDict[(r, c)]
                # 使用OpenCV中的line 函数 在原图中画直线
                cv2.line(I, points[0], points[len(points)-1], (255, ), 2)
    cv2.imshow('accumulator', grayAccu)

    # 显示原图
    cv2.imshow("I", I)
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
