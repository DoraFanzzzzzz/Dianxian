import numpy as np

# 输入数据
'''
loss = np.random.uniform(1, 4, size=5)
active_number = np.array([2, 4, 5, 3, 2])
data = np.array([loss, active_number])
'''
data = np.array([30, 88.7, 3.7, 6.6],[1,2,3,4])
print(data)


# 定义熵值法函数
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 正向化指标
    # x = (x - np.max(x, axis=1).reshape((2, 1))) / (np.max(x, axis=1).reshape((2, 1)) - np.min(x, axis=1).reshape((2, 1)))
    # 反向化指标
    x = (np.max(x, axis=1).reshape((2, 1)) - x) / (
                np.max(x, axis=1).reshape((2, 1)) - np.min(x, axis=1).reshape((2, 1)))

    # 计算第i个研究对象某项指标的比重
    Pij = x / np.sum(x, axis=1).reshape((2, 1))
    ajs = []
    # 某项指标的熵值e
    for i in Pij:
        for j in i:
            if j != 0:
                a = j * np.log(j)
                ajs.append(a)
            else:
                a = 0
                ajs.append(a)
    ajs = np.array(ajs).reshape((2, 5))
    e = -(1 / np.log(5)) * np.sum(ajs, axis=1)
    # 计算差异系数
    g = 1 - e
    # 给指标赋权，定义权重
    w = g / np.sum(g, axis=0)
    return w


w = cal_weight(data)
print(w)