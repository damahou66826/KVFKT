import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 模拟训练数据
m, n = 3, 6
num_samples = 100

# 生成随机的二维矩阵作为输入数据
X_train = np.random.rand(num_samples, m, n)

# 假设输出是每个输入矩阵所有元素的和
y_train = np.sum(X_train, axis=(1, 2))

# 打印一些样本
for i in range(3):
    print(f"样本 {i + 1}:\n输入矩阵:\n{X_train[i]}\n标签: {y_train[i]}\n")

# 创建MLP模型
model = models.Sequential()
model.add(layers.Flatten(input_shape=(m, n)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 打印模型结构
model.summary()

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
