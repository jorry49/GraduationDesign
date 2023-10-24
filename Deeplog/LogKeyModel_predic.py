import argparse
import json

import torch
import torch.nn as nn

# 设定设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name, window_size):
    """
    从数据文件中生成日志序列。

    参数:
        name (str): 数据文件名。
        window_size (int): 日志窗口大小。

    返回:
        set: 包含日志序列的集合。
    """
    hdfs = set()
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def main():
    # Hyperparameters
    num_classes = 28
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    # 模型加载
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 数据加载, 这里使用您的 'generate' 函数从测试集获取数据
    test_normal_loader = generate('hdfs_test_normal', window_size)
    test_abnormal_loader = generate('hdfs_test_abnormal', window_size)

    # 初始化统计变量
    TP = 0  # 真正例
    FP = 0  # 假正例
    FN = 0  # 假反例
    TN = 0  # 真反例

    # 对正常数据进行预测
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label in predicted:
                    TN += 1
                else:
                    FP += 1

    # 对异常数据进行预测
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label in predicted:
                    FN += 1
                else:
                    TP += 1

    # 计算评价指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # recall也被称为真阳性率（TPR）
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算真阳性率（TPR）和假阳性率（FPR）
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # 真阳性率也就是召回率
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 假阳性率

    metrics = {
        'Confusion Matrix': {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        },
        'Precision': precision,
        'Recall': recall,  # recall is TPR
        'F1-score': F1,
        'True Positive Rate': TPR,  # 添加真阳性率
        'False Positive Rate': FPR  # 添加假阳性率
    }

    # 写入到JSON文件中
    with open('evaluation_metrics.json', 'w') as json_file:
        json.dump(metrics, json_file)

    print('Evaluation completed. Metrics:', metrics)


# 当脚本直接运行时，调用main函数
if __name__ == '__main__':
    main()
