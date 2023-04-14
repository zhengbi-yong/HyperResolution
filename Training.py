import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorly import random
from torch.utils.data import DataLoader, random_split

from DataLoader import HyperResolutionDataLoader
from HRTNN import HRTNN
from SRCNN import SRCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available():
    print("cuda is available.")
    # tensor = random.random_tensor(shape=(10, 10, 10), device='cuda', dtype=tl.float32)
else:
    print("we have to use cpu.")
    
root = "./dataset"
source_folder = "LF"
target_folder = "GT"
dataset = HyperResolutionDataLoader(image_root_path=root,
                                        image_source_folder=source_folder,
                                        image_target_folder=target_folder)
train_size = int(0.8 * len(dataset)) # 计算训练集的大小
valid_size = len(dataset) - train_size # 计算验证集的大小
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size]) # 随机划分数据集
train_loader = DataLoader(train_dataset, batch_size=32) # 创建训练集的dataloader
valid_loader = DataLoader(valid_dataset, batch_size=32) # 创建验证集的dataloader
# print(len(train_dataset))
# print(len(valid_dataset))
# print(train_size)
# print(valid_size)

epochs = 100 # 定义训练的轮数
learning_rate = 0.001 # 定义学习率
# model = SRCNN()
model = HRTNN()
model.to(device)
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate) # 创建优化器，使用Adam算法
criterion = nn.MSELoss() # 创建损失函数，使用均方误差

def psnr(img1 , img2):
    # img1和img2是两张同尺寸的灰度图像，形状为(1 , 1 , H , W)
    mse = torch.mean((img1 - img2) ** 2) # 计算两张图像之间的均方误差
    if mse == 0: # 如果均方误差为0，说明两张图像完全相同
        return 100 # 返回一个很大的数值作为PSNR
    max_pixel = 255.0 # 定义最大像素值
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)) # 根据公式计算PSNR
    return psnr.item() # 返回PSNR的数值

train_loss_list = []
train_psnr_list = []
valid_loss_list = []
valid_psnr_list = []
for epoch in range(epochs): # 遍历每一轮训练
    print(f"epoch:{epoch}")
    train_loss = 0.0 # 初始化训练损失为0
    train_psnr = 0.0 # 初始化训练PSNR为0
    for batch in train_loader:
        source, target = batch["source_image"],batch["target_image"]
        source = source.to(device)
        target = target.to(device)
        
        # print(type(source))
        # print(source)
        output = model(source)
        output.to(device)
        optimizer.zero_grad()# 清空梯度
        output = torch.squeeze(output)
        loss = criterion(output , target)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        train_psnr += psnr(output , target)
    train_loss /= len(train_loader)# 计算平均训练损失
    train_psnr /= len(train_loader)# 计算平均训练PSNR
    
    valid_loss = 0.0
    valid_psnr = 0.0
    for batch in valid_loader:
        source, target = batch["source_image"],batch["target_image"]
        source = source.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()# 清空梯度
        output = model(source)
        output.to(device)
        output = torch.squeeze(output)
        loss = criterion(output , target)
        # loss.backward(retain_graph=True)
        # optimizer.step()
        valid_loss += loss.item()
        valid_psnr += psnr(output , target)
    valid_loss /= len(valid_loader)
    valid_psnr /= len(valid_loader)
    print(f"train_loss:{train_loss}")
    print(f"train_psnr:{train_psnr}")
    print(f"valid_loss:{valid_loss}")
    print(f"valid_psnr:{valid_psnr}")
    train_loss_list.append(train_loss)
    train_psnr_list.append(train_psnr)
    valid_loss_list.append(valid_loss)
    valid_psnr_list.append(valid_psnr)
x_list = list(range(epochs))
plt.plot(x_list, train_loss_list, 'b*--', alpha=0.5, linewidth=1, label='train_loss')#'
plt.plot(x_list, train_psnr_list, 'rs', alpha=0.5, linewidth=1, label='train_psnr')
plt.plot(x_list, valid_loss_list, 'go--', alpha=0.5, linewidth=1, label='valid_loss')
plt.plot(x_list, valid_psnr_list, 'g*', alpha=0.5, linewidth=1, label='valid_psnr')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss/psnr')
# plt.show()
plt.savefig("HRTNN2.png")
model_file_name = f"HRTNN2.pth"
torch.save(model.state_dict(), model_file_name)  