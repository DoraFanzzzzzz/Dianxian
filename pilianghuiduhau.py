'''
import numpy as np
from PIL import Image
import cv2
# 打开彩色图片
color_image = Image.open("C:/Users/Dora/Desktop/1.jpg")

# 将彩色图片转换为灰度图片
gray_image = color_image.convert('L')
weighted_mean_img = np.dot(color_image, [0.299, 0.587, 0.114])
# 保存灰度图片


cv2.imwrite(C:/Users/Dora/Desktop/11.jpg,img_to_save)
'''
import os
from PIL import Image
import cv2
import numpy as np
# 设置输入和输出文件夹的路径
input_dir = "C:/Users/Dora/Desktop/in"  # 输入文件夹路
output_dir = "C:/Users/Dora/Desktop/out"  # 输出文件夹路径

# 如果输出文件夹不存在，则创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 循环遍历输入文件夹中的所有PNG文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # 打开彩色图像
        color_image = Image.open(os.path.join(input_dir, filename))

        # 将彩色图像转换为灰度图像
        gray_image = color_image.convert('L')
       # weighted_mean_img = np.dot(color_image, [0.299, 0.587, 0.114])
       # weighted_mean_img = np.dot(color_image, [0.114, 0.587, 0.299])
        # 将灰度图像保存到输出文件夹中
        gray_image.save(os.path.join(output_dir, filename))
        #weighted_mean_img.save(os.path.join(output_dir, filename))


'''

'''