import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def data_load(train_file_name, test_file_name):
    train_data = pd.read_csv(train_file_name, header=None)
    test_data = pd.read_csv(test_file_name, header=None)
    return train_data, test_data

# shuffle
def data_shuffle(train_data, test_data):
    np.random.seed(0)  # set a random seed to replay
    index = np.arange(train_data.shape[0])
    np.random.shuffle(index)
    train_data = train_data.iloc[index]
    train_label = train_data.iloc[:, -1]
    train_data = train_data.iloc[:, 24:-1]
    test_label = test_data.iloc[:, -1]
    test_data = test_data.iloc[:, 24:-1]

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':

    train_file_name = r'data/train1.csv'
    test_file_name = r'data/test.csv'
    train_data, test_data = data_load(train_file_name, test_file_name)
    train_data, train_label, test_data, test_label = data_shuffle(train_data, test_data)

    # 计算每个特征与目标变量之间的 Pearson 相关系数
    correlations = []
    for feature in train_data.columns:
        corr, _ = pearsonr(train_data[feature], train_label)
        correlations.append(corr)

    # 创建包含特征和相关系数的 DataFrame
    df = pd.DataFrame({'Feature': train_data.columns, 'Correlation': correlations})

    # 根据相关系数排序特征重要性
    df = df.sort_values(by='Correlation', ascending=False)

    # 打印特征重要性排序结果
    print(df)