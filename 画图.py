import matplotlib.pyplot as plt
import numpy as np

# 生成一组正态分布的数据
mu, sigma = 0, 0.1  # 均值和标准差
data = np.random.normal(mu, sigma, 1000)

# 绘制正态分布图
plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Normal Distribution')
plt.show()
