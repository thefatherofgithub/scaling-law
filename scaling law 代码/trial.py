import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便复现结果
np.random.seed(40)
debug = 1
# 参数
d = 100  # 向量维度
N = 5   # 类别数
M = 20  # 参数量
T = 10000  # 样本数量
learning_rate = 3e-4  # 学习率
num_epochs = 1  # 训练轮数

# 随机生成 W* 矩阵 (真实权重矩阵)
W_star = np.random.randn(N, d)

# 交叉熵损失函数
def cross_entropy_loss(logits, y):
    logits -= np.max(logits, axis=1, keepdims=True)  # 数值稳定性处理
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[np.arange(len(y)), y])
    loss = np.mean(correct_logprobs)
    return loss, probs

# # 训练使用 SGD
def train(S, V, M, T, num_epochs, learning_rate):

    # 生成 T 个 d 维高斯分布向量 x
    # 生成对应的标签 y = argmax_i <w_i, x>
    X = np.random.randn(T, d)
    y = np.argmax(np.dot(X, W_star.T), axis=1)

    losses = []
    mid = np.dot(X, S.T)
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(T):
            mid_i = mid[i:i+1]  # 取第 i 个样本
            y_i = y[i]  # 对应的标签

            # 计算损失和梯度
            loss, probs = cross_entropy_loss(np.dot(mid_i, V), np.array([y_i]))
            total_loss += loss

            # 计算梯度
            grad = probs
            grad[0, y_i] -= 1  # 针对正确类别的梯度调整
            dV = np.dot(mid_i.T, grad)  # 计算权重的梯度

            # 随机梯度下降更新
            V -= learning_rate * dV

        # 记录每轮损失
        avg_loss = total_loss / T
        losses.append(avg_loss)
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return V, losses, X, y

# 下面是一个训练尝试
S = np.random.randn(M, d)
V = np.random.randn(M, N)
_, losses, _, _ = train(S, V, M, T, num_epochs, learning_rate)
# 绘制训练损失
plt.plot(range(num_epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss Over Time")
plt.show()

if(debug == 0):
    # Scaling law 图像
    scaling_laws = []
    T_values = [500, 1000, 5000]  # 不同的样本数 T
    M_values = [3, 5, 10]  # 不同的参数量 M

    for M in M_values:
        
        # 初始化 S, V 矩阵 (需要优化的权重矩阵)
        S = np.random.randn(M, d)
        V = np.random.randn(M, N)

        for T in T_values:
        
            V_final, _, X, y = train(S, V, M, T, num_epochs, learning_rate)
            
            # 计算最后的损失并记录
            final_loss, _ = cross_entropy_loss(np.dot(np.dot(X, S.T), V_final), y)
            scaling_laws.append((T, M, final_loss))

    scaling_laws = np.array(scaling_laws)

    # 绘制 Scaling Law 图像
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # 图像 1: 随着 T 变化的 Loss

    for M_val in M_values:
        mask = (scaling_laws[:, 1] == M_val)
        ax[0].plot(scaling_laws[mask, 0], scaling_laws[mask, 2], label=f'M={M_val}')
    ax[0].set_xlabel("Sample size T")
    ax[0].set_ylabel("Final Loss")
    ax[0].legend()
    ax[0].set_title("Loss Scaling with T")

        # 图像 2: 随着 M 变化的 Loss

    for T_val in T_values:
        mask = (scaling_laws[:, 0] == T_val)
        ax[1].plot(scaling_laws[mask, 1], scaling_laws[mask, 2], label=f'T={T_val}')
    ax[1].set_xlabel("Number of Classes N")
    ax[1].set_ylabel("Final Loss")
    ax[1].legend()
    ax[1].set_title("Loss Scaling with N")

    plt.tight_layout()
    plt.show()
