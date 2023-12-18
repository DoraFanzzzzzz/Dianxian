import cv2
import os

# 设置高斯核大小和标准差
kernel_size = (7, 7)
sigma = 0.5

# 遍历指定目录下所有灰度图像文件
for root, dirs, files in os.walk('C:/Users/Dora/Desktop/out1'):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            # 读取灰度图像
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            # 高斯滤波
            img = cv2.GaussianBlur(img, kernel_size, sigma)
            # 保存处理后的图像文件
            cv2.imwrite(os.path.join('C:/Users/Dora/Desktop/out2', file), img)