import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from data_preparation import prepare_data

def tune_model():
    # 获取训练集数据
    X_train, _ = prepare_data()
    # 计算样本数量
    num_samples = len(X_train)
    # 划分前半部分为正样本，后半部分为负样本
    half_num = num_samples // 2
    y_train = [1] * half_num + [0] * (num_samples - half_num)

    # 定义超参数网格
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # 创建 SVM 分类器
    model = SVC()

    # 创建 GridSearchCV 对象
    grid_search = GridSearchCV(model, param_grid, cv=5)

    # 进行网格搜索
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和最佳得分
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    best_model = tune_model()