import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class CPR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CPR2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pad = nn.ZeroPad2d((0, 0, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # 先卷积再填充
        x = self.conv(x)
        x = self.pad(x)
        x = self.relu(x)

        # 实际上大多数是先填充再卷积
        # x = self.conv(x)
        # x = self.pad(x)
        # x = self.relu(x)
        return x


class CRP1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, stride):
        super(CRP1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 2D CNN layers
        self.cpr2d_1 = CPR2d(3, 32, kernel_size=(3,128))
        self.cpr2d_2 = CPR2d(32, 32, kernel_size=(3,1))
        self.cpr2d_3 = CPR2d(32, 32, kernel_size=(3,1))
        self.cpr2d_4 = CPR2d(32, 32, kernel_size=(3,1))

        # 1D CNN layers
        self.crp1d_1 = CRP1d(32, 32, 6, 3, 2)
        self.crp1d_2 = CRP1d(32, 32, 6, 3, 2)

        # Fully connected layer
        self.fc = nn.Linear(864, 2)

    def forward(self, x0):
        # 2D CNN forward pass
        x1 = self.cpr2d_1(x0)
        x2 = self.cpr2d_2(x1)
        x3 = self.cpr2d_3(x2)
        x4 = self.cpr2d_4(x3)

        x = x1 + x2 + x3 + x4
        #print("1", x.size())
        x = x.squeeze(-1)
        #print("2", x.size())
        # 1D CNN forward pass
        x = self.crp1d_1(x)
        #print("3", x.size())
        x = self.crp1d_2(x)
        #print("4", x.size())
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# # Example usage
# model = CustomCNN()
# input_image = torch.randn(1, 3, 32, 32)  # Example input
# output = model(input_image)
# print(output.shape)  # Output shape
