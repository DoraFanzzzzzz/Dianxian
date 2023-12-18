import cv2
import os

# 设置输入路径和输出路径
input_path = "C:/Users/Dora/Desktop/out2"
output_path = "C:/Users/Dora/Desktop/out3"

# 如果输出文件夹不存在，则创建输出文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 获取待处理图片列表
file_list = os.listdir(input_path)

# 循环处理每张图片
for f in file_list:
    # 读入图片并转为灰度图像
    img = cv2.imread(os.path.join(input_path, f), cv2.IMREAD_GRAYSCALE)

    # 进行Canny边缘检测
    edge = cv2.Canny(img, 150, 220)

    # 保存处理后的图片
    cv2.imwrite(os.path.join(output_path, f), edge)
