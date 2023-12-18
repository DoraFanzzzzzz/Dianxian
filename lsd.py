import cv2

# 读取图像
img = cv2.imread('C:/Users/Dora/Desktop/1.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行 LSD 操作
params = {'scale': 0.2, 'sigma_scale': 1.1, 'quant': 8.0, 'ang_th': 22.5, 'log_eps': 0.0, 'density_th': 0.5}
lsd = cv2.createLineSegmentDetector(**params).detect(gray)[0]

# 绘制直线
for line in lsd:
    x1, y1, x2, y2 = map(int, line[0])
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
