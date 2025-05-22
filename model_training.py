from sklearn.svm import SVC
from data_preparation import prepare_data

def train_model():
    X_train, _ = prepare_data()
    # 计算样本数量
    num_samples = len(X_train)
    # 划分前半部分为正样本，后半部分为负样本
    half_num = num_samples // 2
    y_train = [1] * half_num + [0] * (num_samples - half_num)

    # 创建 SVM 分类器
    model = SVC(kernel='linear')

    # 训练模型
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    model = train_model()
    print("模型训练完成")