import torch
import torch.nn as nn
import time
import argparse

# 设备配置，检查是否有GPU可用，如果有则使用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成函数，从文件中读取日志数据并生成会话序列
def generate(name):
    hdfs = set()  # 使用集合来存储数据，以加速测试和去重
    with open('Data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

# 深度学习模型定义
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()

        # 设置模型中的参数
        self.hidden_size = hidden_size  # LSTM隐藏层的大小
        self.num_layers = num_layers    # LSTM的层数
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 创建一个LSTM层，它将序列数据作为输入，并具有指定数量的层
        # input_size：输入数据的特征大小
        # hidden_size：隐藏状态的大小
        # num_layers：LSTM层的数量
        # batch_first=True 表示输入数据的形状为(batch_size, sequence_length, input_size)

        self.fc = nn.Linear(hidden_size, num_keys)
        # 创建一个全连接层，用于输出模型的预测结果
        # hidden_size 是 LSTM 层的输出大小，num_keys 是模型要预测的类别数量

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 初始化LSTM的初始隐藏状态h0和细胞状态c0为零张量
        # x.size(0) 是输入张量的批次大小，self.num_layers 表示 LSTM 的层数
        # self.hidden_size 表示每个隐藏状态的大小

        out, _ = self.lstm(x, (h0, c0))
        # 将输入序列x和初始状态(h0, c0)传递给LSTM层
        # out 是LSTM层的输出，包含了序列中每个时间步的隐藏状态信息
        # _ 是LSTM层的最终隐藏状态，通常不使用在这里

        out = self.fc(out[:, -1, :])
        # 使用全连接层对LSTM的输出进行线性变换
        # out[:, -1, :] 表示从LSTM输出张量中选择最后一个时间步的信息作为预测

        return out

if __name__ == '__main__':
    # 超参数配置
    num_classes = 28  # 模型的输出类别数量，这里有28个不同的类别

    input_size = 1  # 每个时间步的输入特征维度，这里设置为1，表示每个时间步的输入是一个标量

    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'
    # 预训练模型的文件路径，模型将从此路径加载预训练的权重参数

    parser = argparse.ArgumentParser()  # 创建命令行参数解析器的实例

    parser.add_argument('-num_layers', default=2, type=int)
    # 添加一个命令行参数：num_layers，如果未提供，默认值为2，参数值的类型是整数

    parser.add_argument('-hidden_size', default=64, type=int)
    # 添加一个命令行参数：hidden_size，如果未提供，默认值为64，参数值的类型是整数

    parser.add_argument('-window_size', default=10, type=int)
    # 添加一个命令行参数：window_size，如果未提供，默认值为10，参数值的类型是整数

    parser.add_argument('-num_candidates', default=9, type=int)
    # 添加一个命令行参数：num_candidates，如果未提供，默认值为9，参数值的类型是整数

    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在args变量中

    num_layers = args.num_layers  # 从命令行参数中获取num_layers的值并赋给变量
    hidden_size = args.hidden_size  # 从命令行参数中获取hidden_size的值并赋给变量
    window_size = args.window_size  # 从命令行参数中获取window_size的值并赋给变量
    num_candidates = args.num_candidates  # 从命令行参数中获取num_candidates的值并赋给变量

    # 创建模型实例并加载预训练模型参数
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # 生成测试数据集（正常和异常）
    test_normal_loader = generate('hdfs_test_normal')
    test_abnormal_loader = generate('hdfs_test_abnormal')

    # 初始化True Positives（TP）和False Positives（FP）的计数
    TP = 0
    FP = 0

    # 测试模型性能
    start_time = time.time()

    # 遍历正常数据集
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                # 选择最高的候选事件
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                # 如果真实标签不在候选事件中，则增加FP计数
                if label not in predicted:
                    FP += 1
                    break

    # 遍历异常数据集
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                # 选择最高的候选事件
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                # 如果真实标签不在候选事件中，则增加TP计数
                if label not in predicted:
                    TP += 1
                    break

    # 计算运行时间
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    # 计算性能指标，包括False Positives（FP）、False Negatives（FN）、Precision（精确度）、Recall（召回率）和F1度量
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')
