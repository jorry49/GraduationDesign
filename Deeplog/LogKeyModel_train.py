import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# 设备配置，检查是否有GPU可用，如果有则使用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 函数：从文件中读取日志数据并生成会话序列
def generate_sessions(data_filename):
    num_sessions = 0  # 初始化会话数量为0，用于统计会话的数量
    inputs = []  # 初始化一个空列表 inputs，用于存储输入序列数据
    outputs = []  # 初始化一个空列表 outputs，用于存储输出序列数据

    # 打开名为 data_filename 的文件，通常用于存储日志数据的文件
    with open('Data/' + data_filename, 'r') as f:
        # 遍历文件中的每一行数据
        for line in f.readlines():
            num_sessions += 1  # 每读取一行数据，会话数量加1，用于统计会话的数量

            # 将读取的一行数据按空格分割，将分割后的字符串转换为整数，
            # 然后将每个整数减去1，最后将这些整数组成一个元组 session
            session = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            # 遍历 session 中的元素，生成输入序列和输出序列
            for i in range(len(session) - window_size):
                # 将当前位置到当前位置加上窗口大小 window_size 的元素切片作为输入序列，
                # 并添加到 inputs 列表中
                inputs.append(session[i:i + window_size])

                # 将当前位置加上窗口大小 window_size 后的元素作为输出序列，
                # 并添加到 outputs 列表中
                outputs.append(session[i + window_size])

    # 打印读取的会话数量，这个信息会在生成数据集时显示出来
    print('会话数量({}): {}'.format(data_filename, num_sessions))

    # 打印生成的序列数量，这个信息会在生成数据集时显示出来
    print('序列数量({}): {}'.format(data_filename, len(inputs)))

    # 将生成的输入序列和输出序列转换为 PyTorch 的 Tensor，并使用 TensorDataset 封装成一个数据集 dataset
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))

    # 返回生成的数据集，包含了输入序列和输出序列
    return dataset

# 深度学习模型定义
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def main():
    # 超参数配置
    num_classes = 28  # 模型的输出类别数量，这里有28个不同的类别
    num_epochs = 300  # 训练模型的迭代次数，也称为"周期"
    batch_size = 2048  # 每个训练迭代中用于更新模型的样本数量
    input_size = 1  # 输入数据的特征维度，这里设置为1，表示每个时间步的输入是一个标量
    model_dir = 'model'  # 模型保存的文件夹路径，训练后的模型将保存在这个文件夹中
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))

    # 创建命令行参数解析器的实例
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()

    # 从命令行参数中获取 num_layers、hidden_size 和 window_size 的值
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    # 创建模型实例并将其移动到GPU（如果可用）
    model = SequenceModel(input_size, hidden_size, num_layers, num_classes).to(device)

    # 生成训练数据集
    seq_dataset = generate_sessions('hdfs_train')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 使用TensorBoard进行可视化
    writer = SummaryWriter(log_dir='log/' + log)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # 循环多次遍历数据集
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # 前向传播
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('第 [{}/{}] 轮, 训练损失: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('运行时间: {:.3f}s'.format(elapsed_time))

    # 保存模型参数
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, log + '.pt')
    torch.save(model.state_dict(), model_path)
    print('模型已保存到:', model_path)
    writer.close()
    print('训练完成')


if __name__ == '__main__':
    main()
