# 如果当前脚本被执行，而不是被其他脚本导入，则运行以下代码
if __name__ == '__main__':
    # 初始化几个空列表，用于存储从文件中读取的日志序列
    hdfs_train = []
    hdfs_test_normal = []
    hdfs_test_abnormal = []

    # 初始化三个集合，用于存储唯一的事件（这里称为模板）
    h1 = set()
    h2 = set()
    h3 = set()

    # 以读模式打开训练数据文件
    with open('data/hdfs_train', 'r') as f:
        # 读取文件的每一行
        for line in f.readlines():
            # 移除行尾的换行符，按空格分割，将分割结果（字符串）转换为整数，并减去1，然后将其转换为元组
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            # 将处理后的行数据（元组）添加到训练列表中
            hdfs_train.append(line)

    # 处理训练数据，提取所有独特的事件到集合h1中
    for line in hdfs_train:
        for c in line:
            h1.add(c)  # 添加到集合中，自动处理重复事件

    # 以下部分与上述处理训练数据的代码相似，但是用于处理正常的测试数据
    with open('data/hdfs_test_normal', 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_normal.append(line)
    for line in hdfs_test_normal:
        for c in line:
            h2.add(c)

    # 以下部分与上述处理训练数据的代码相似，但是用于处理异常的测试数据
    with open('data/hdfs_test_abnormal', 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            hdfs_test_abnormal.append(line)
    for line in hdfs_test_abnormal:
        for c in line:
            h3.add(c)

    # 打印训练集的统计信息：总行数和不同的模板（事件）数
    print('train length: %d, template length: %d, template: %s' % (len(hdfs_train), len(h1), h1))
    # 打印正常测试集的统计信息：总行数和不同的模板（事件）数
    print('test_normal length: %d, template length: %d, template: %s' % (len(hdfs_test_normal), len(h2), h2))
    # 打印异常测试集的统计信息：总行数和不同的模板（事件）数
    print('test_abnormal length: %d, template length: %d, template: %s' % (len(hdfs_test_abnormal), len(h3), h3))
