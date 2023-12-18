'''
import numpy as np

# 根据指标值生成评价矩阵
criteria = np.array([
    [20, 10000, 200],
    [15, 9000, 220],
    [12, 6000, 180],
    [18, 12000, 250],
])

# 计算每个指标的平均值和标准差
criteria_mean = np.mean(criteria, axis=0)
criteria_std = np.std(criteria, axis=0)

# 计算每个指标的变异系数
criteria_cv = criteria_std / criteria_mean

# 计算每个指标的critic值
criteria_critic = criteria_cv / np.sum(criteria_cv)

# 输出结果
print("评价矩阵：")
print(criteria)

print("每个指标的平均值：")
print(criteria_mean)

print("每个指标的标准差：")
print(criteria_std)

print("每个指标的变异系数：")
print(criteria_cv)

print("每个指标的critic值：")
print(criteria_critic)

print("客观权重：")
print(np.sum(criteria_critic))
'''

import numpy as np
'''
# 根据指标值生成评价矩阵
criteria = np.array([
    [20, 10000, 200],
    [15, 9000, 220],
    [12, 6000, 180],
    [18, 12000, 250],
])
'''
criteria = np.array([
    [1, 3, 5, 6],
    [1/3, 1, 3, 5],
    [1/5, 1/3, 1, 3],
    [1/6, 1/5, 1/3, 1]
])
# 计算每个指标的加权平均值
criteria_weighted_mean = np.mean(criteria, axis=0)

# 计算每个指标之间的矩阵关系
criteria_diff = criteria_weighted_mean - criteria
criteria_matrix = np.matmul(np.transpose(criteria_diff), criteria_diff)

# 计算每个指标与加权平均值之间的差异
criteria_diff = criteria - criteria_weighted_mean

# 计算每个指标的方差
criteria_var = np.var(criteria, axis=0)

# 计算每个指标的CRITIC值
criteria_critic = np.sum(criteria_matrix / (criteria_var[:, None] * criteria_var[None, :]), axis=1)
criteria_critic /= np.sum(criteria_critic)

# 输出结果
print("评价矩阵：")
print(criteria)

print("每个指标的加权平均值：")
print(criteria_weighted_mean)

print("每个指标之间的矩阵关系：")
print(criteria_matrix)

print("每个指标与加权平均值之间的差异：")
print(criteria_diff)

print("每个指标的方差：")
print(criteria_var)

print("每个指标的CRITIC值：")
print(criteria_critic)

print("客观权重：")
print(np.sum(criteria_critic))
