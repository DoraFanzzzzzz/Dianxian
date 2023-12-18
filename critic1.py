#完整代码
#导入相关库
import pandas as pd
import numpy as np
#导入数据
data=pd.read_excel('C:/Users/Dora/Desktop/1.xlsx')
label_need=data.keys()[1:]
data1=data[label_need].values
#查看行数和列数
data2 = data1
[m,n]=data2.shape
#负向指标标准化
index=[2] #负向指标位置,注意python是从0开始计数，对应位置也要相应减1
for j in index:
    d_max=max(data1[:,j])
    d_min=min(data1[:,j])
    data2[:,j]=(d_max-data1[:,j])/(d_max-d_min)
# 正向指标标准化
#正向指标位置
index_all=np.arange(n)
index=np.delete(index_all,index) 
for j in index:
    d_max=max(data1[:,j])
    d_min=min(data1[:,j])
    data2[:,j]=(data1[:,j]-d_min)/(d_max-d_min)
#对比性
the=np.std(data2,axis=0)
#矛盾性
data3=list(map(list,zip(*data2))) #矩阵转置
r=np.corrcoef(data3)   #求皮尔逊相关系数
f=np.sum(1-r,axis=1)
#信息承载量
c=the*f
#计算权重
w=c/sum(c)
#计算得分
s=np.dot(data2,w)
Score=100*s/max(s) 
for i in range(0,len(Score)):
    print(f"{data['银行'][i]}银行百分制评分为：{Score[i]}")  