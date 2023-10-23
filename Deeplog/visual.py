# 导入numpy库，用于数组和矩阵计算
# 导入matplotlib的pyplot模块，用于绘图
import matplotlib.pyplot as plt
import numpy as np

# 定义两组分数，每组三个值，分别代表精确度、召回率和F1分数
PCA = (0.98, 0.67, 0.79)
LSTM = (0.9526, 0.9903, 0.9711)

# 创建一个新的画布和一个轴对象
fig, ax = plt.subplots()

# 定义条形图的一些参数
index = np.arange(3)  # 定义三个分类
bar_width = 0.2  # 条形的宽度
opacity = 0.4  # 条形的透明度

# 在轴上绘制两组数据的条形图
rects1 = ax.bar(index, PCA, bar_width, alpha=opacity, color='b', label='PCA')  # 绘制PCA的条形图
rects2 = ax.bar(index + bar_width, LSTM, bar_width, alpha=opacity, color='r', label='LSTM')  # 绘制LSTM的条形图

# 设置图表的标签和标题
ax.set_xlabel('Measure')  # X轴标签，表示评估指标
ax.set_ylabel('Scores')  # Y轴标签，表示得分
ax.set_title('Scores by different models')  # 图表的总标题
ax.set_xticks(index + bar_width / 2)  # 设置X轴刻度位置
ax.set_xticklabels(('Precesion', 'Recall', 'F1-score'))  # 设置X轴刻度标签
ax.legend()  # 显示图例

# 显示图形
plt.show()

# 定义一些数据点和计算精确度、召回率和F1分数所需的参数
x = [8, 9, 10, 11]  # 窗口大小
FP = [605, 588, 495, 860]  # 假正例数
FN = [465, 333, 108, 237]  # 假负例数
TP = [4123 - FN[i] for i in range(4)]  # 真正例数，通过计算得出
# 下面计算精确度、召回率和F1分数
P = [TP[i] / (TP[i] + FP[i]) for i in range(4)]  # 精确度计算公式
R = [TP[i] / (TP[i] + FN[i]) for i in range(4)]  # 召回率计算公式
F1 = [2 * P[i] * R[i] / (P[i] + R[i]) for i in range(4)]  # F1分数计算公式

# 使用不同的标记和线条样式绘制精确度、召回率和F1分数
l1 = plt.plot(x, P, ':rx')  # 绘制精确度，红色虚线，x标记
l2 = plt.plot(x, R, ':b+')  # 绘制召回率，蓝色虚线，+标记
l3 = plt.plot(x, F1, ':k^')  # 绘制F1分数，黑色虚线，上三角标记

# 设置X轴和Y轴的标签
plt.xlabel('window_size')  # X轴标签，表示窗口大小
plt.ylabel('scores')  # Y轴标签，表示得分

# 显示图例
plt.legend((l1[0], l2[0], l3[0]), ('Precision', 'Recall', 'F1-score'))

# 设置X轴的刻度
plt.xticks(x)

# 设置Y轴的范围
plt.ylim((0.5, 1))

# 显示图形
plt.show()
