import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preparation import prepare_data
from model_training import train_model

def evaluate_model():
    # 获取训练集和测试集数据
    X_train, X_test = prepare_data()

    # 为训练集和测试集创建标签
    num_train_samples = len(X_train)
    half_train_num = num_train_samples // 2
    y_train = [1] * half_train_num + [0] * (num_train_samples - half_train_num)

    num_test_samples = len(X_test)
    half_test_num = num_test_samples // 2
    y_test = [1] * half_test_num + [0] * (num_test_samples - half_test_num)

    # 训练模型
    model = train_model()

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

if __name__ == "__main__":
    evaluate_model()