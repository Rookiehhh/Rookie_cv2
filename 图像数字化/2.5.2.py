"""
将RGB 彩色图转换为三维的ndarray
"""
import cv2
import numpy as np
import sys

if __name__ == '__main__':
    img = cv2.imread(f"SWMF-GM-Meridian-P-Forecast-20170318180000.jpg")
    cv2.imshow("Color Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
