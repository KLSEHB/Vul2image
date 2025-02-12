import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time
import os
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms

TrainFilePath = '/home/liao/projects/codetranslate_linux/Pic/Pic_150_all_23248/'
RealFilePath = '/home/liao/projects/Vul2Image_Large-scale_image_dataset/'
# RealFilePath = '/home/liao/projects/codetranslate_linux/Pic_150_all_22918_apply/'
ModelSavePath = '/home/liao/projects/model.pth'
# NegativeImgPath = '/home/liao/projects/real_negative_image/'
num_epochs = 100
learning_rate = 0.0003
train_batch_size = 256
apply_batch_size = 512
threshold = 1.00001


# 定义模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(64 * 34 * 34, 81)  # 需要根据图像尺寸调整
        self.fc2 = nn.Linear(81, 2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 训练和评估模型
def train(model, X_train, Y_train, batch_size, epochs):
    # 数据加载器
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # 训练模型
    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")



def apply(model, X_real, img_paths, batch_size):

    model.eval()
    model.to(device)

    dataset = TensorDataset(X_real)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    negative_samples = []
    negative_paths = []
    with torch.no_grad():
        # for idx, batch in enumerate(tqdm(dataloader, desc="The model is running predictions")):
        #     inputs = batch[0]
        #     inputs = inputs.to(device)
        #     inputs = inputs.permute(0, 3, 1, 2)
        #     outputs = model(inputs)
        #     _, predicted = torch.max(outputs, 1)
        #
        #     for i, prediction in enumerate(predicted):
        #         if prediction == 0:
        #             negative_samples.append(X_real[idx * dataloader.batch_size + i])
        #             negative_paths.append(img_paths[idx * dataloader.batch_size + i])

        global threshold
        for idx, batch in enumerate(tqdm(dataloader, desc="The model is running predictions")):
            inputs = batch[0]
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)  # 假设已经使用softmax
            probabilities = torch.softmax(outputs, dim=1)  # 如果没有在模型最后进行softmax，则在这里应用

            for i, prob in enumerate(probabilities):
                if prob[0] < threshold:  # 类别0的概率低于阈值
                    # 不认为它是类别0
                    continue
                else:
                    negative_samples.append(X_real[idx * dataloader.batch_size + i])
                    negative_paths.append(img_paths[idx * dataloader.batch_size + i])



    # 如果需要保存负类图像，可以如下操作
    # for i, img in enumerate(negative_samples):
    #     negative_img = Image.fromarray(img)
    #     negative_img.save(f"{NegativeImgPath}negative_sample_{i}.bmp")


    with open('negative_samples.txt', 'w') as f:
        for path in negative_paths:
            f.write(f"{path}\n")

    print(f"{len(negative_paths)} negative samples were identified")



def LoadFile(FilePath):
    image = Image.open(FilePath)
    # print("finish\n")
    return np.array(image)

def GetDirAllFile(FilePath):
    train_x = []
    train_y = []
    for root, dirs, files in os.walk(FilePath):
        for file in files:
            path = os.path.join(root, file)
            if '_0.bmp' in path:
                train_y.append(0)
            elif '_1.bmp' in path:
                train_y.append(1)
            x = LoadFile(path)

            train_x.append(x)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def GetRealDirAllFile(FilePath):
    real_x = []
    img_paths = []
    for root, dirs, files in os.walk(FilePath):
        for file in files:
            path = os.path.join(root, file)
            img_arry = LoadFile(path)

            img_paths.append(path)
            real_x.append(img_arry)

    real_x = np.array(real_x)

    return real_x, img_paths

def train_model():
    X, Y = GetDirAllFile(TrainFilePath)#x为图片的numpy数组 y为标签的numpy数组
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y, dtype=torch.long)

    model = CNNModel()
    train(model, X_train_tensor, Y_train_tensor, train_batch_size, num_epochs)
    torch.save(model.state_dict(), ModelSavePath)
    print(f"The model has been successfully saved to {ModelSavePath}")

def apply_model():

    X_real, img_paths = GetRealDirAllFile(RealFilePath)
    X_real_tensor = torch.tensor(X_real, dtype=torch.float32)


    model = CNNModel()
    model.load_state_dict(torch.load(ModelSavePath))
    T1 = time.time()
    apply(model, X_real_tensor, img_paths, apply_batch_size)
    T2 = time.time()
    print("cost time:", T2 - T1)

def main(action):

    if action == 'train':
        train_model()
    elif action == 'apply':
        apply_model()
    else:
        print("未知的操作，请输入 'train' 或 'apply'")

if __name__=='__main__':

    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description='训练或应用模型')
    parser.add_argument('action', type=str, choices=['train', 'apply'], help='指定是否训练模型或应用已训练的模型')
    args = parser.parse_args()
    main(args.action)
