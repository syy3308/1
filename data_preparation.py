import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # 读取 CSV 文件
    data = pd.read_csv('data/features_all.csv')

    # 提取特征和标签（这里假设第一列是文件名，后面的列是特征）
    X = data.iloc[:, 1:].values
    # 由于这里只是人脸检测，暂时不考虑标签，如果是分类任务，需要根据实际情况设置标签

    # 划分训练集和测试集
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test

if __name__ == "__main__":
    X_train, X_test = prepare_data()
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")





