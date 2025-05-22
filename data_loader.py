import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedShuffleSplit

class FaceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 1].values
        try:
            self.labels = self.labels.astype(int)
        except ValueError as e:
            print(f"转换标签为整数时出错: {e}")
            raise
        self.features = self.data.drop(self.data.columns[1], axis=1).iloc[:, 1:].values
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)

        num_classes = len(np.unique(self.labels))
        if num_classes > 1:
            try:
                ada = ADASYN()
                self.features, self.labels = ada.fit_resample(self.features, self.labels)
            except ValueError:
                print("少数类样本数量过少，无法使用ADASYN进行过采样。")
        else:
            print("标签只有一个类别，不进行过采样。")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def get_data_loaders(batch_size=32):
    dataset = FaceDataset('data/features_all.csv')
    all_features = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
    all_labels = np.array([dataset[i][1].numpy() for i in range(len(dataset))])

    class_counts = np.bincount(all_labels)
    if np.min(class_counts) < 2:
        print("某个类别样本数量小于2，使用简单随机抽样。")
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(all_features, all_labels):
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
