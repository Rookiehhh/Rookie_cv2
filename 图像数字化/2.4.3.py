"""
将灰度图转换为 ndarray
"""
import sys
import cv2
import numpy as np

if __name__ == '__main__':
    # 输入图像矩阵， 转换为array
    img = cv2.imread(f"SWMF-GM-Meridian-P-Forecast-20170318180000.jpg", cv2.IMREAD_GRAYSCALE)
    print(type(img))
    print(img)
    # 显示图像
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
