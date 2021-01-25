import numpy as np
import matrixslow as ms

"""
制造训练样本
1. 根据均值为171，标准差为6的正态分布采样500个男性身高
2. 根据均值为158，标准差为5的正态分布采样500个女性身高
3. 根据均值为70，标准差为10的正态分布采样500个男性体重
4. 根据均值为57，标准差为8的正态分布采样500个女性体重
5. 根据均值为16，标准差为2的正态分布采样500个男性体脂率
6. 根据均值为22，标准差为2的正态分布采样500个女性体脂率
7. 构造500个1，作为男性标签
8. 构造500个-1，作为女性标签

将以上所有数据构成一个 1000 x 4 的NumPy数组，每一行为 (身高，体重，体脂率，性别标签)
"""

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

np.random.shuffle(train_set)  # 打乱样本顺序

"""
构造计算图: 输入向量是 3 x 1 矩阵, 无需初始化，不参与训练
"""
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

"""
性别标签: 其中 1 表示男，-1 表示女
"""
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

"""
权重向量: 1 x 3 矩阵，需要初始化(自动初始化为正态分布随机抽取的值), 参与训练
"""
w = ms.core.Variable(dim = (1, 3), init = True, trainable=True)

"""
阈值: 1 x 1 矩阵，需要初始化，参与训练
"""
b = ms.core.Variable(dim = (1, 1), init=True, trainable=True)

# Adaline的预测输出