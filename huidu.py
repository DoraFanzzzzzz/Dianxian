
import numpy as np
import matplotlib.pyplot as plt
def run(img_path):
    img = plt.imread(img_path)
    print("原图: ", img.shape)
    max_img = img.max(axis=2)
    mean_img = img.mean(axis=2)
    weighted_mean_img = np.dot(img, [0.299, 0.587, 0.114])
    titles = ["img", "max_img", u'mean_img', u'weighted_mean_img']
    images = [img, max_img, mean_img, weighted_mean_img]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.savefig('C:/Users/Dora/Desktop/11.jpg')
    plt.show()
    plt.savefig('C:/Users/Dora/Desktop/test.jpg')
if __name__ == '__main__':
    run("C:/Users/Dora/Desktop/1.jpg")
    pass



'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("C:/Users/Dora/Desktop/1.jpg")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(img_gray[2,1])
# 二值化
img_binary = np.where(img_gray/255 >= 0.5, 1, 0)

# 画图
img = plt.imread("C:/Users/Dora/Desktop/1.jpg")
plt.subplot(221)

plt.imshow(img)
plt.title('Color map')

plt.subplot(222)
plt.savefig('C:/Users/Dora/Desktop/11.jpg') #必须要放在show()前面，show()之后会自动释放图表内存
plt.imshow(img_gray, cmap='gray')
plt.title('gray')

plt.subplot(223)
plt.savefig('C:/Users/Dora/Desktop/11.jpg')
plt.imshow(img_binary, cmap='gray')
plt.title('img_binary')
plt.savefig('mytestplt.png')
plt.show()

'''