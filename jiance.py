import os
import cv2



input_folder = "C:/Users/Dora/Desktop/in"
output_folder = "C:/Users/Dora/Desktop/out5"
# 根据图片的特征，提取输电线路
def extract_power_line(img):
    # TODO: 根据图片特征，使用图像分割方法将输电线路从背景中提取出来
    return power_line_img

# 确认异物坐标位置
def detect_abnormal_region(img):
    # TODO: 使用区域生长的方法对图像再分割，以像素点均值比较的方法确认异物坐标位置
    abnormal_region_coord = [100, 100, 200, 200]  # TODO: 假设此处得到的坐标为 [x_min, y_min, x_max, y_max]
    return abnormal_region_coord

# 在原始图像中将异常处框出并输出相对坐标位置
def draw_rect_and_save(img_path, abnormal_region_coord):
    img = cv2.imread(img_path)
    x_min, y_min, x_max, y_max = abnormal_region_coord
    # 在图像中标记异常处
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=2)
    # 输出相对坐标位置
    height, width, _ = img.shape
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_relative = x_center / width
    y_relative = y_center / height
    print('文件名: {}, 异常处相对坐标: ({:.2f}, {:.2f})'.format(os.path.basename(img_path), x_relative, y_relative))
    # 保存结果
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img)

# 对输入文件夹中的所有图片进行处理
for file in os.listdir(input_folder):
    if file.endswith('.jpg') or file.endswith('.png'):
        img_path = os.path.join(input_folder, file)
        # 提取输电线路
        power_line_img = extract_power_line(cv2.imread(img_path))
        # 确认异物坐标位置
        abnormal_region_coord = detect_abnormal_region(power_line_img)
        # 在原始图像中将异常处框出并输出相对坐标位置
        draw_rect_and_save(img_path, abnormal_region_coord)
