

import cv2
import numpy as np
import os

# 设置输入路径和输出路径
input_path = "C:/Users/Dora/Desktop/in"
output_path = "C:/Users/Dora/Desktop/out5"

# 循环处理每张图片
for f in os.listdir(input_path):
    # 读入图片并转为灰度图像
    img = cv2.imread(os.path.join(input_path, f), cv2.IMREAD_GRAYSCALE)

    # 进行Canny边缘检测
    edges = cv2.Canny(img, 150, 200)

    # 进行霍夫直线变换
    lines = cv2.HoughLines(edges, 1, np.pi / 150, threshold=220)

    # 在原始图像上绘制直线
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 保存处理后的图片
    cv2.imwrite(os.path.join(output_path, f), color_img)