import torch
import torch.nn as nn


class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3,  padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.batch1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.pool(self.relu(self.batch2(self.conv2(x))))
        x = self.dropout(x)
        x = self.adpool(x).squeeze(-1)
        
        x = self.relu(self.fc1(x))
        #x = x.transpose(1,0)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class StreamingCNNArch(nn.Module):
    def __init__(self, channel1, channel2):
        super().__init__()

        self.conv1 = nn.Conv1d(1, channel1, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(channel1, channel2, kernel_size=3)
        self.fc1 = nn.Linear(channel2, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.adpool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        #x = x.transpose(1,0)
        x = self.fc2(x)
        x = x.squeeze()
        return x



class NoBatchAudioClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3,  padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.adpool(x).squeeze()
        
        
        x = self.relu(self.fc1(x))
        #x = x.transpose(1,0)
        x = self.fc2(x)
        x = x.squeeze()
        return x