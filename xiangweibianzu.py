import cv2
import numpy as np
import os

# 设置输入路径和输出路径
input_path = "C:/Users/Dora/Desktop/in"
output_path = "C:/Users/Dora/Desktop/out5"

# 获取待处理图片列表
file_list = os.listdir(input_path)

# 相位编组算法参数
n_theta = 30  # 方向数量
theta_max = np.pi# 方向范围（弧度）
r_min = 5 # 编组半径（像素）
kernel_size = (7, 7)
sigma = 1.0
# 循环处理每张图片
for f in file_list:
    # 读入图片并转为灰度图像
    img = cv2.imread(os.path.join(input_path, f), cv2.IMREAD_GRAYSCALE)

    img = cv2.GaussianBlur(img, kernel_size, sigma)

    # 进行Canny边缘检测，得到边缘点的坐标
    edges = cv2.Canny(img, 150, 220)
    points = np.argwhere(edges > 0)

    # 计算每个点的幅值和相位
    r = np.sqrt(np.sum(np.power(points, 2), axis=1))
    theta = np.arctan2(points[:, 0], points[:, 1])

    # 对每个点进行相位编组
    bins = np.zeros((n_theta, len(points)))
    for i in range(n_theta):
        indices = np.where((theta >= (i / n_theta) * theta_max) & (theta < ((i + 1) / n_theta) * theta_max))[0]
        r_bin = r[indices]
        indices = indices[np.where(r_bin >= r_min)]
        bins[i][indices] = 1

    # 对每个编组进行直线拟合
    lines = []
    for i in range(n_theta):
        sample_points = points[bins[i] > 0]
        if len(sample_points) < 2:
            continue
        xs = sample_points[:, 1]
        ys = sample_points[:, 0]
        A = np.vstack([xs, np.ones(len(xs))]).T
        k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        lines.append((k, b))

    # 在原始图像上绘制直线
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        k, b = line
        x0 = 0
        y0 = int(b)
        x1 = img.shape[1]
        y1 = int(k * x1 + b)
        cv2.line(color_img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    # 保存处理后的图片
    cv2.imwrite(os.path.join(output_path, f), color_img)
