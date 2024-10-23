from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
import random
from tqdm import tqdm
np.random.seed(40)
debug = 1
d = 100  # 向量维度
N = 5   # 类别数
M = 20  # 参数量
compute = 200  # 计算成本
lr = 3e-4  # 学习率
j = 1 
i = 1
S = torch.randn(d, M, dtype = torch.float64)
W_star_crloss = np.random.randn(d, N)
W_star_mseloss = np.random.randn(d, 1)
class Net(nn.Module):
    def __init__(self, d, N, M):
        super().__init__()
        self.S = S    
        self.V = nn.Linear(M, N, bias=False)
    def forward(self, input):
        f1 = torch.matmul(input, self.S)
        f2 = self.V(f1)
        return f2
    def train(self, input, y, epoch, learning_rate, criterion):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        loss_history = []
        data_num = input.size()[0]
        for j in range(epoch):
            for i in torch.randperm(data_num):
                optimizer.zero_grad()
                output = self(input)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())            
        return loss_history
epoch_num = np.array([1, 5, 10, 20, 40, 50])

# cross_entropy_loss, 比较single epoch和multiple epoch的效果
plt.figure()
plt.title("cr_loss: single epoch vs multiple epoch")
plt.xlabel("data number")
plt.ylabel("loss")
for epoch in tqdm(epoch_num):
    model = Net(d, N, M).double()
    data_num = int(compute / epoch)
    X = torch.randn(data_num, d, dtype = torch.float64)
    y = torch.argmax(torch.matmul(X, torch.tensor(W_star_crloss)), dim = 1)
    loss_history = model.train(X, y, epoch, lr, criterion = F.cross_entropy)
    plt.plot(range(1, compute + 1), loss_history, label='epoch=%d' % epoch)
plt.legend()
plt.show()

# MSEloss, 比较single epoch和multiple epoch的效果
plt.figure()
plt.title("mse_loss: single epoch vs multiple epoch")
plt.xlabel("data number")
plt.ylabel("loss")
for epoch in tqdm(epoch_num):
    model = Net(d, 1, M).double()
    data_num = int(compute / epoch)
    X = torch.randn(data_num, d, dtype = torch.float64)
    y = torch.matmul(X, torch.tensor(W_star_mseloss))
    loss_history = model.train(X, y, epoch, lr, criterion = F.mse_loss)
    plt.plot(range(1, compute + 1), loss_history, label='epoch=%d' % epoch)
plt.legend()
plt.show()

# cross_entropy_loss, 比较single epoch和multiple epoch的效果, 加入noise
plt.figure()
plt.title("cr_loss: single epoch vs multiple epoch + noise")
plt.xlabel("data number")
plt.ylabel("loss")
for epoch in tqdm(epoch_num):
    model = Net(d, N, M).double()
    data_num = int(compute / epoch)
    X = torch.randn(data_num, d, dtype = torch.float64)
    prob_mat = F.softmax(torch.matmul(X, torch.tensor(W_star_crloss)))
    y = torch.multinomial(prob_mat, num_samples = 1, replacement=True).view(-1)
    loss_history = model.train(X, y, epoch, lr, criterion = F.cross_entropy)
    plt.plot(range(1, compute + 1), loss_history, label='epoch=%d' % epoch)
plt.legend()
plt.show()

# MSEloss, 比较single epoch和multiple epoch的效果, 加入noise
plt.figure()
plt.title("mse_loss: single epoch vs multiple epoch")
plt.xlabel("data number")
plt.ylabel("loss")
for epoch in tqdm(epoch_num):
    model = Net(d, 1, M).double()
    data_num = int(compute / epoch)
    X = torch.randn(data_num, d, dtype = torch.float64)
    y = torch.matmul(X, torch.tensor(W_star_mseloss)) + torch.randn(data_num, 1, dtype = torch.float64)
    loss_history = model.train(X, y, epoch, lr, criterion = F.mse_loss)
    plt.plot(range(1, compute + 1), loss_history, label='epoch=%d' % epoch)
plt.legend()
plt.show()
# cross_entropy_loss的scaling law

# 绘制关于M的scaling law：
compute = 1000
M_values = range(1, 21)
pbar = tqdm(M_values, desc="j={} i={}".format(j, i), ncols = 80)
plt.figure()
plt.title("Scaling Law for M: Compute = {}".format(compute))
final_loss = []
for M in pbar:
    S = torch.randn(d, M, dtype = torch.float64)   
    model = Net(d, N, M).double()  
    data_num = compute
    X = torch.randn(data_num, d, dtype = torch.float64)
    y = torch.argmax(torch.matmul(X, torch.tensor(W_star_crloss)), dim = 1)
    loss_history = model.train(X, y, 1, lr, criterion = F.cross_entropy)
    final_loss.append(loss_history[compute-1])
plt.xlabel("M")
plt.ylabel("loss")
plt.xticks(range(2, 21, 2))
plt.plot(M_values, final_loss)
plt.show()

# 绘制关于compute的scaling law：
compute_values = range(100, 2100, 100)
M = 10
plt.figure()
plt.title("Scaling Law for Compute: M = {}".format(M))
final_loss = []
S = torch.randn(d, M, dtype = torch.float64) 
for compute in tqdm(compute_values, ncols = 80):
    model = Net(d, N, M).double()  
    data_num = compute
    X = torch.randn(data_num, d, dtype = torch.float64)
    y = torch.argmax(torch.matmul(X, torch.tensor(W_star_crloss)), dim = 1)
    loss_history = model.train(X, y, 1, lr, criterion = F.cross_entropy)
    final_loss.append(loss_history[compute-1])
plt.xlabel("compute")
plt.ylabel("loss")
# plt.xticks(range(200, 2100, 200))
# plt.xscale("log")
# plt.yscale("log")
plt.plot(compute_values, final_loss)
plt.show()

    # # 不同的参数量 M
    # T_values = [500, 1000, 5000]  # 不同的样本数 T
    # grid = [(M, T) for M in M_values for T in T_values]
    # X = torch.randn(data_num, d, dtype = torch.float64)
    # y = torch.argmax(torch.matmul(X, torch.tensor(W_star_crloss)), dim=1)

# torch.tensor是float32，而torch.randn和torch.nn.Linear 是float64，所以会报错