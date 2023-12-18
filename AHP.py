
'''
import numpy as np

# 评价矩阵
criteria = np.array([
    [1, 1.8, 1.7*1.8, 1.6*1.7*1.8],
    [1/1.8, 1, 1/1.3, 1/(1.3*1.4)],
    [1/(1.7*1.8), 1.3, 1, 1/1.2],
    [1/(1.6*1.7*1.8), 1.4*1.3, 1.2, 1]
])

# 计算每个指标对应的平均值
criteria_mean = np.mean(criteria, axis=0)

# 计算归一化后的矩阵
criteria_norm = criteria / criteria_mean[:, None]

# 计算每个指标的权重
criteria_weight = np.mean(criteria_norm, axis=0)

# 输出结果
print("评价矩阵：")
print(criteria)

print("归一化后的矩阵：")
print(criteria_norm)

print("每个指标的权重：")
print(criteria_weight)

'''

import numpy as np
'''
judgment_matrix = np.array([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]])
'''
# 标度构造法结果
judgment_matrix = np.array([
    [1, 2, 5],
    [1/2, 1, 4],
    [1/5, 1/4, 1],

])

# 计算特征向量
eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)

# 提取最大特征值所对应的特征向量
max_eigenvalue_index = np.argmax(eigenvalues)
weights = eigenvectors[:, max_eigenvalue_index].real
weights /= weights.sum()

# 输出每个评价对象的权重
for i, weight in enumerate(weights):
    print(f"评价对象 {i+1} 的权重：{weight:.2%}")


import numpy as np
'''
judgment_matrix = np.array([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]])
'''
# 标度构造法结果
A = np.array([
    [1, 3, 5],
    [1 / 3, 1, 4],
    [1 / 5, 1 / 4, 1],
])

# 定义原矩阵 A 和 n

n = 3

# 对 A 的每一行元素累乘，得到一个新的列向量 B
B = np.prod(A, axis=1).reshape(-1, 1)

# 将 B 的每个分量开 n 次方
B_n = np.power(B, 1/n)

# 对 B_n 进行归一化，得到权重向量 w
w = B_n / np.sum(B_n)

print(w)