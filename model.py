import torch
import torch.nn as nn


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        # 第一个全连接层，输入维度128，输出维度64
        self.fc1 = nn.Linear(128, 64)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层，输入维度64，输出维度2
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # 通过第一个全连接层
        x = self.fc1(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x


class ImprovedFaceCNN(nn.Module):
    def __init__(self):
        super(ImprovedFaceCNN, self).__init__()
        # 第一个全连接层，输入维度128，输出维度512
        self.fc1 = nn.Linear(128, 512)
        # 一维批量归一化层
        self.bn1 = nn.BatchNorm1d(512)
        # ReLU激活函数
        self.relu1 = nn.ReLU()
        # 随机丢弃50%的神经元
        self.dropout1 = nn.Dropout(0.5)

        # 第二个全连接层，输入维度512，输出维度256
        self.fc2 = nn.Linear(512, 256)
        # 一维批量归一化层
        self.bn2 = nn.BatchNorm1d(256)
        # ReLU激活函数
        self.relu2 = nn.ReLU()
        # 随机丢弃50%的神经元
        self.dropout2 = nn.Dropout(0.5)

        # 第三个全连接层，输入维度256，输出维度128
        self.fc3 = nn.Linear(256, 128)
        # 一维批量归一化层
        self.bn3 = nn.BatchNorm1d(128)
        # ReLU激活函数
        self.relu3 = nn.ReLU()
        # 随机丢弃50%的神经元
        self.dropout3 = nn.Dropout(0.5)

        # 第四个全连接层，输入维度128，输出维度64
        self.fc4 = nn.Linear(128, 64)
        # 一维批量归一化层
        self.bn4 = nn.BatchNorm1d(64)
        # ReLU激活函数
        self.relu4 = nn.ReLU()
        # 随机丢弃50%的神经元
        self.dropout4 = nn.Dropout(0.5)

        # 第五个全连接层，输入维度64，输出维度2
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        # 通过第一个全连接层
        x = self.fc1(x)
        # 应用批量归一化
        x = self.bn1(x)
        # 应用ReLU激活函数
        x = self.relu1(x)
        # 随机丢弃部分神经元
        x = self.dropout1(x)

        # 通过第二个全连接层
        x = self.fc2(x)
        # 应用批量归一化
        x = self.bn2(x)
        # 应用ReLU激活函数
        x = self.relu2(x)
        # 随机丢弃部分神经元
        x = self.dropout2(x)

        # 通过第三个全连接层
        x = self.fc3(x)
        # 应用批量归一化
        x = self.bn3(x)
        # 应用ReLU激活函数
        x = self.relu3(x)
        # 随机丢弃部分神经元
        x = self.dropout3(x)

        # 通过第四个全连接层
        x = self.fc4(x)
        # 应用批量归一化
        x = self.bn4(x)
        # 应用ReLU激活函数
        x = self.relu4(x)
        # 随机丢弃部分神经元
        x = self.dropout4(x)

        # 通过第五个全连接层
        x = self.fc5(x)
        return x


class SimplifiedFaceCNN(nn.Module):
    def __init__(self):
        super(SimplifiedFaceCNN, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x