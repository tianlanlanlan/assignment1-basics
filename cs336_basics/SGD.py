import numpy as np
import matplotlib.pyplot as plt


# Taken from: https://www.ibm.com/think/topics/stochastic-gradient-descent
def stochastic_gradient_descent(X, y, lr=0.01, epochs=1000, tol=1e-6):
    # Initialize parameters randomly
    w = np.random.randn()
    b = np.random.randn()
    n = len(X)
    losses = []

    prev_loss = float("inf")

    for epoch in range(epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)
        # print(f"indices = {indices}")

        for i in indices:
            xi = X[i]
            yi = y[i]

            # Prediction
            y_pred = xi * w + b

            # Comppute gradients (derivates)
            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)

            # Update parameters
            w -= lr * dw
            b -= lr * db

        # Compute loss at the end of the epoch
        loss = np.mean((y - (w * X + b)) ** 2)
        losses.append(loss)
        if abs(prev_loss - loss) < tol:
            print(f"Stop early at epoch {epoch + 1}, with loss = {loss}")
            break

        prev_loss = loss

    return w, b, losses


# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成 1000 个样本，特征 X 为均匀分布在 [0, 10] 的数值
X = np.random.rand(1000) * 10

# 设定真实的参数：w = 2.5, b = 1.3
true_w = 2.5
true_b = 1.3

# 生成目标值 y，并添加一些高斯噪声（标准差为 2）
y = true_w * X + true_b + np.random.randn(1000) * 2

pred_w, pred_b, total_losses = stochastic_gradient_descent(
    X, y, lr=0.01, epochs=100, tol=1e-6
)
print(f"Estimate w = {pred_w}, b = {pred_b}. true w = {true_w}, true b = {true_b}")

# Plot image
plt.plot(total_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
