import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def load_metrics(file_path):
    """
    从JSON文件中加载评估指标数据。

    参数:
        file_path (str): JSON文件的路径。

    返回:
        dict: 包含评估指标的字典。
    """
    with open(file_path, 'r') as file:
        metrics = json.load(file)
    return metrics


def plot_confusion_matrix(confusion_matrix, class_names, font, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵的函数。

    参数:
        confusion_matrix (dict): 包含混淆矩阵的字典。
        class_names (list): 类别名称的列表。
        font: 字体属性。
        title (str): 图表标题。
        cmap: 颜色映射。
    """
    cm = np.array([
        [confusion_matrix['TN'], confusion_matrix['FP']],
        [confusion_matrix['FN'], confusion_matrix['TP']]
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Labels',
           xlabel='Predicted Labels')

    set_font_properties(ax, font)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontproperties=font)

    fig.tight_layout()
    plt.show()


def plot_metrics(metrics, font):
    """
    绘制性能指标的柱状图。

    参数:
        metrics (dict): 包含性能指标的字典。
        font: 字体属性。
    """
    performance_metrics = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['Precision'], metrics['Recall'], metrics['F1-score']]

    bar_chart(performance_metrics, metrics_values, ['blue', 'green', 'orange'],
              'Performance Metrics', 'Value', 'Model Performance Metrics', font)


def bar_chart(x_labels, y_values, colors, x_label, y_label, chart_title, font):
    """
    绘制柱状图的通用函数。

    参数:
        x_labels (list): X轴标签。
        y_values (list): Y轴值。
        colors (list): 柱子颜色。
        x_label (str): X轴标签。
        y_label (str): Y轴标签。
        chart_title (str): 图表标题。
        font: 字体属性。
    """
    plt.bar(x_labels, y_values, color=colors)
    plt.xlabel(x_label, fontproperties=font)
    plt.ylabel(y_label, fontproperties=font)
    plt.title(chart_title, fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.show()


def set_font_properties(ax, font):
    """
    设置轴的字体属性。

    参数:
        ax: Matplotlib轴对象。
        font: 字体属性。
    """
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontproperties=font)
    plt.setp(ax.get_yticklabels(), fontproperties=font)


def main():
    metrics_file_path = 'evaluation_metrics.json'
    font_path = r"C:\Windows\Fonts\msyh.ttc"  # 替换为您的字体文件路径
    font = FontProperties(fname=font_path)

    metrics = load_metrics(metrics_file_path)

    confusion_matrix = metrics["Confusion Matrix"]

    plot_confusion_matrix(confusion_matrix, ['Normal', 'Abnormal'], font)

    print(f"Precision: {metrics['Precision']}")
    print(f"Recall: {metrics['Recall']}")
    print(f"F1 Score: {metrics['F1-score']}")

    # 生成性能指标的柱状图
    plot_metrics(metrics, font)


if __name__ == "__main__":
    main()
