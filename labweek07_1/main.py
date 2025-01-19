import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.001, n_iterations=2000):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.losses = []

    def sign(self, x):
        return np.where(x > 0, 1, -1)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            loss = 0
            for i in range(X.shape[0]):
                prediction = self.sign(np.dot(X[i], self.weights) + self.bias)
                if prediction != y[i]:
                    loss -= y[i] * (np.dot(X[i], self.weights) + self.bias)
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
            self.losses.append(loss)

    def predict(self, X):
        return self.sign(np.dot(X, self.weights) + self.bias)


# 读取 CSV 文件
data = pd.read_csv('data.csv')

# 提取特征和标签
X = data[['Age', 'EstimatedSalary']].values
y = np.where(data['Purchased'].values == 1, 1, -1)  # 将标签从0, 1转换为-1, 1

# 归一化特征
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 实例化感知机对象并进行训练
perceptron = Perceptron()
perceptron.fit(X, y)

# 预测
predictions = perceptron.predict(X)

# 计算分类准确率
accuracy = np.mean(predictions == y)

# 输出参数和准确率
print("Learning rate:", perceptron.learning_rate)
print("Num of iterations:", perceptron.n_iterations)
print("Classification Accuracy: {:.4%}".format(accuracy))

# 绘制数据可视化图
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.title('Data Visualization')
plt.colorbar(label='Purchased')
plt.grid(False)
plt.show()

# 绘制损失曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(perceptron.losses) + 1), perceptron.losses, marker='o', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(False)
plt.show()

# 超平面
w1,w2=perceptron.weights
b=perceptron.bias
x_line = np.array([X[:, 0].min(),X[:, 0].max()])
y_line = (-w1/w2)*x_line-(b/w2)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')
plt.plot(x_line, y_line,color='red', linestyle='--',label='Decision Boundary')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.title('Planoid Visualization')
plt.colorbar(label='Purchased')
plt.grid(False)
plt.show()
