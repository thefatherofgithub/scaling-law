# 引入必要的库
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
# 设置随机种子
torch.manual_seed(42)

# 配置参数
d = 100  # 向量维度
N = 5    # 类别数
M = 20  # 参数量
compute = 12000  # 计算成本
lr_1 = 1e-4 # 学习率
lr_2 = 2e-5
eps = 1  # mse_loss 的标准差
j = 1
i = 1
debug = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 加速

# 初始化权重矩阵
S = 1 / math.sqrt(M) * torch.randn(d, M, dtype=torch.float32).to(device)
W_star_crloss = torch.randn(d, N, dtype=torch.float32).to(device)
W_star_mseloss = torch.randn(d, 1, dtype=torch.float32).to(device)
epoch_num = torch.tensor([1, 5, 10, 20, 40, 60, 80, 100], dtype=torch.int32)
batch_value = torch.tensor([1], dtype=torch.int32)
# 生成数据函数
def generate_data(d, data_num, f, isrand):
    # 生成随机数据 X
    X = torch.randn(int(data_num), d, dtype=torch.float32).to(device)
    
    # 生成对应的标签 y
    if f == F.cross_entropy:
        W_star = W_star_crloss
        if isrand == 0:
            y = torch.argmax(torch.matmul(X, W_star), dim=1)
        else:
            y = torch.multinomial(F.softmax(torch.matmul(X, W_star), dim=1), num_samples=1, replacement=True).view(-1)
    elif f == F.mse_loss:
        W_star = W_star_mseloss
        if isrand == 0:
            y = torch.matmul(X, W_star)
        else:
            y = torch.matmul(X, W_star) + eps * torch.randn(int(data_num), 1).to(device)
    return X, y

# 构建神经网络
class Net(nn.Module):
    def __init__(self, d, N, M):
        super().__init__()
        self.S = S  # 初始化 S 矩阵
        self.V = nn.Linear(M, N, bias=False).to(device)  # 初始化线性层
        
    def forward(self, input):
        f1 = torch.matmul(input, self.S)  # 计算 f1
        f2 = self.V(f1)  # 计算 f2
        return f2
    
    def test(self, X_val, y_val, criterion):
        with torch.no_grad():
            output = self(X_val)
            loss = criterion(output, y_val)
            return loss.item()
    
    def train(self, X_train, y_train, X_val, y_val, epoch, learning_rate, criterion, compute_loss, batch_size = 1):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        loss_history = []
        data_num = X_train.size()[0]
        for j in range(epoch):
            for i in range(0, data_num, batch_size):
                optimizer.zero_grad()
                output = self(X_train[i:i+batch_size])
                loss = criterion(output, y_train[i:i+batch_size])
                loss.backward()
                optimizer.step()
                if compute_loss == 1:
                   loss_history.append(self.test(X_val, y_val, criterion))
        return loss_history

# 示例：cross_entropy_loss，比较 single epoch 和 multiple epoch 的效果

plt.figure()
plt.title("cr_loss: single epoch vs multiple epoch")
plt.xlabel("data number")
plt.ylabel("loss")
for batch_size in batch_value:
    for epoch in tqdm(epoch_num):
        model = Net(d, N, M).to(device)
        data_num = int(compute / epoch.item())
        X_train, y_train = generate_data(d, data_num, F.cross_entropy, 0)
        X_val, y_val = generate_data(d, data_num // 5, F.cross_entropy, 0)
        loss_history = model.train(X_train, y_train, X_val, y_val, epoch.item(), lr_1, criterion=F.cross_entropy, compute_loss=1, batch_size=batch_size)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='epoch=%d' % epoch.item())
    plt.legend()
    plt.show()

# 示例：mse_loss，比较 single epoch 和 multiple epoch 的效果

plt.figure()
plt.title("mse_loss: single epoch vs multiple epoch")
plt.xlabel("data number")
plt.ylabel("loss")

for batch_size in batch_value:
    for epoch in tqdm(epoch_num):    
        model = Net(d, 1, M).to(device)
        data_num = int(compute / epoch.item())
        X_train, y_train = generate_data(d, data_num, F.mse_loss, 0)
        X_val, y_val = generate_data(d, data_num // 5, F.mse_loss, 0)
        loss_history = model.train(X_train, y_train, X_val, y_val, epoch.item(), lr_2, criterion=F.mse_loss, compute_loss=1, batch_size=batch_size)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='epoch=%d' % epoch.item())
    plt.legend()
    plt.show()

# 绘制 Cross Entropy Loss，比较单次训练和多次训练的效果，并加入噪声

plt.figure()
plt.title("Cross Entropy Loss: Single vs. Multiple Epochs + Noise")
plt.xlabel("Data Number")
plt.ylabel("Loss")
for batch_size in batch_value:
    for epoch in tqdm(epoch_num):
        model = Net(d, N, M).to(device)
        data_num = int(compute / epoch.item())
        X_train, y_train = generate_data(d, data_num, F.cross_entropy, 1)
        X_val, y_val = generate_data(d, data_num // 5, F.cross_entropy, 1)
        loss_history = model.train(X_train, y_train, X_val, y_val, epoch.item(), lr_1, criterion=F.cross_entropy, compute_loss=1, batch_size=batch_size)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='epoch=%d' % epoch.item())
    plt.legend()
    plt.show()

# 绘制 MSE Loss，比较单次训练和多次训练的效果，并加入噪声

plt.figure()
plt.title("MSE Loss: Single vs. Multiple Epochs + Noise")
plt.xlabel("Data Number")
plt.ylabel("Loss")
for batch_size in batch_value:
    for epoch in tqdm(epoch_num):
        model = Net(d, 1, M).to(device)
        data_num = int(compute / epoch.item())
        X_train, y_train = generate_data(d, data_num, F.mse_loss, 1)
        X_val, y_val = generate_data(d, data_num // 5, F.mse_loss, 1)
        loss_history = model.train(X_train, y_train, X_val, y_val, epoch.item(), lr_2, criterion=F.mse_loss, compute_loss=1, batch_size=batch_size)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='epoch=%d' % epoch.item())
    plt.legend()
    plt.show()


# 绘制 Cross Entropy Loss 关于 M 的 Scaling Law
compute = 1000
M_values = range(1, 21)
plt.figure()
plt.title("Cross Entropy Loss Scaling Law for M")
plt.xlabel("M")
plt.ylabel("Loss")
final_loss = []
for M in tqdm(M_values, desc="Scaling Law for M"):
    S = 1 / math.sqrt(M) * torch.randn(d, M, dtype=torch.float32).to(device)
    model = Net(d, N, M).to(device)
    data_num = compute
    X_train, y_train = generate_data(d, data_num, F.cross_entropy, 0)
    X_val, y_val = generate_data(d, data_num // 5, F.cross_entropy, 0)
    model.train(X_train, y_train, X_val, y_val, 1, lr_1, criterion=F.cross_entropy, compute_loss=0)
    final_loss.append(F.cross_entropy(model(X_val), y_val).item())
plt.plot(M_values, final_loss, marker='o')
plt.xticks(range(2, 21, 2))
plt.show()

# 绘制 MSE Loss 关于 M 的 Scaling Law
compute = 1000
M_values = range(1, 21)
plt.figure()
plt.title("MSE Loss Scaling Law for M")
plt.xlabel("M")
plt.ylabel("Loss")
final_loss = []
for M in tqdm(M_values, desc="Scaling Law for M"):
    S = 1 / math.sqrt(M) * torch.randn(d, M, dtype=torch.float32).to(device)
    model = Net(d, 1, M).to(device)
    data_num = compute
    X_train, y_train = generate_data(d, data_num, F.mse_loss, 0)
    X_val, y_val = generate_data(d, data_num // 5, F.mse_loss, 0)
    model.train(X_train, y_train, X_val, y_val, 1, lr_2, criterion=F.mse_loss, compute_loss=0)
    final_loss.append(F.mse_loss(model(X_val), y_val).item())
plt.plot(M_values, final_loss, marker='o')
plt.xticks(range(2, 21, 2))
plt.show()

# 绘制 Cross Entropy Loss 关于 compute 的 Scaling Law
compute_values = range(100, 2100, 100)
M = 10
plt.figure()
plt.title("Cross Entropy Loss Scaling Law for Compute: M = {}".format(M))
final_loss = []
S = 1 / math.sqrt(M) * torch.randn(d, M, dtype=torch.float32).to(device)  # 使用 float32，并放置在设备上
for compute in tqdm(compute_values, ncols=80):
    model = Net(d, N, M).to(device)  # 将模型移动到 GPU 上
    data_num = compute
    X_train, y_train = generate_data(d, data_num, F.cross_entropy, 0)
    X_val, y_val = generate_data(d, data_num // 5, F.cross_entropy, 0)
    model.train(X_train, y_train, X_val, y_val, 1, lr_1, criterion=F.cross_entropy, compute_loss=0)
    final_loss.append(F.cross_entropy(model(X_val), y_val).item())
plt.xlabel("Compute")
plt.ylabel("Loss")
# # 启用对数刻度绘图（可选），更好地显示数据的变化趋势
# plt.xscale("log")
# plt.yscale("log")
plt.plot(compute_values, final_loss, marker='o')
plt.show()

# 绘制 MSE Loss 关于 compute 的 Scaling Law
compute_values = range(100, 2100, 100)
M = 10
plt.figure()
plt.title("MSE Loss Scaling Law for Compute: M = {}".format(M))
final_loss = []
S = 1 / math.sqrt(M) * torch.randn(d, M, dtype=torch.float32).to(device)  # 使用 float32，并放置在设备上
for compute in tqdm(compute_values, ncols=80):
    model = Net(d, 1, M).to(device)  # 将模型移动到 GPU 上
    data_num = compute
    X_train, y_train = generate_data(d, data_num, F.mse_loss, 0)
    X_val, y_val = generate_data(d, data_num // 5, F.mse_loss, 0)
    model.train(X_train, y_train, X_val, y_val, 1, lr_2, criterion=F.mse_loss, compute_loss=0)
    final_loss.append(F.mse_loss(model(X_val), y_val).item())
plt.xlabel("Compute")
plt.ylabel("Loss")
# # 启用对数刻度绘图（可选），更好地显示数据的变化趋势
# plt.xscale("log")
# plt.yscale("log")
plt.plot(compute_values, final_loss, marker='o')
plt.show()
