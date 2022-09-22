# @Author : yiling
# @File : origin.py
# @Description:感知机原始算法
# @Project: MachineLearning
# @CreateTime : 2022/9/22 20:37:52

# 定义感知机模型类
import numpy as np


class Perceptorn:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_late = 1  # 步长

    # 算法主函数
    def fit(self, x_train, y_train):
        self.w = np.zeros(x_train.shape[1])  # x_train矩阵列数（这里也特指特征或者 维度/？）
        i = 0

        while i < x_train.shape[0]:  # x_train行数（指x的个数）
            x = x_train[i]  # [3,3]  [4,3]  [1,1]
            y = y_train[i]  # 1      1      -1
            # 判断其所有点是否都没有误分类，如有更新w,b,重新判断
            if y * (np.dot(self.w, x) + self.b) <= 0:  # dot()点积，所以w的初始化和x有关
                self.w = self.w + self.l_late * np.dot(x, y)  # 这里因为y是一维的所以相当于一个常数点乘（点积）矩阵，直接用y*x也可以
                self.b = self.b + self.l_late * y
                i = 0
            else:
                i += 1


# 训练集
x_train1 = np.array([[3, 3], [4, 3], [1, 1]])
y_train1 = np.array([1, 1, -1])

# 调用
perceptorn = Perceptorn()
perceptorn.fit(x_train1, y_train1)

# 输出结果
print(perceptorn.w, perceptorn.b)
