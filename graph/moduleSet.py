import tensorflow as tf
import numpy as np

# 启用 Eager Execution 模式
tf.compat.v1.enable_eager_execution()

class FuncsGRU(tf.keras.Model):
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super(FuncsGRU, self).__init__()

        # 表示创建 n_layer的全连接层，输入都为hidden_dim，激活函数为relu
        self.lins = [
            tf.keras.layers.Dense(hidden_dim, activation='relu')
            for _ in range(n_layer)
        ]
        # 丢弃层 dpo表示丢弃神经元的比例
        self.dropout = tf.keras.layers.Dropout(dpo)
        # 输出层 表示输出维度为output_dim
        self.out = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, x):
        for lin in self.lins:
            x = lin(x)
        out_x = self.out(x)
        return tf.add(out_x, self.dropout(out_x)), self.dropout(x)

class MyGRU(tf.keras.Model):
    def __init__(self, n_layer, input_dim, hidden_dim):
        super(MyGRU, self).__init__()

        self.g_ir = FuncsGRU(n_layer, input_dim, hidden_dim, 0)
        self.g_iz = FuncsGRU(n_layer, input_dim, hidden_dim, 0)
        self.g_in = FuncsGRU(n_layer, input_dim, hidden_dim, 0)
        self.g_hr = FuncsGRU(n_layer, hidden_dim, hidden_dim, 0)
        self.g_hz = FuncsGRU(n_layer, hidden_dim, hidden_dim, 0)
        self.g_hn = FuncsGRU(n_layer, hidden_dim, hidden_dim, 0)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, inputs):
        x, h = inputs
        r_t, r_t_dropout = self.g_ir(x)
        z_t, z_t_dropout = self.g_iz(x)
        n_t, n_t_dropout = self.g_in(x)

        print("r_t shape: ", r_t.shape)
        print(self.g_hr(h)[0])

        r_t = self.sigmoid(tf.add(r_t, self.g_hr(h)[0]))
        z_t = self.sigmoid(tf.add(z_t, self.g_hz(h)[0]))
        n_t = self.tanh(tf.add(n_t, tf.multiply(self.g_hn(h)[0], r_t)))

        h_t = tf.add(tf.multiply(1 - z_t, n_t), tf.multiply(z_t, h))
        return h_t, r_t_dropout, z_t_dropout, n_t_dropout

if __name__ == '__main__':
    # 模拟训练数据
    num_samples = 100
    seq_length = 10
    input_dim = 5
    hidden_dim = 8

    # 生成随机的输入序列
    X_train = np.random.rand(num_samples, seq_length, input_dim).astype(np.float32)

    # 初始化隐藏状态
    initial_hidden_state = tf.zeros((num_samples, seq_length, hidden_dim), dtype=tf.float32)
    print("Initial_hidden_state shape: ", initial_hidden_state.shape)

    # 构建 MyGRU 模型
    my_gru = MyGRU(n_layer=3, input_dim=input_dim, hidden_dim=hidden_dim)

    # 编译模型（不适用于 Eager Execution 模式）
    # my_gru.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    for epoch in range(10):
        with tf.GradientTape() as tape:
            predictions = my_gru([X_train, initial_hidden_state])
            loss = tf.losses.mean_squared_error(initial_hidden_state, predictions)

        gradients = tape.gradient(loss, my_gru.trainable_variables)
        optimizer = tf.train.AdamOptimizer()
        optimizer.apply_gradients(zip(gradients, my_gru.trainable_variables))
        print("Epoch {}, Loss: {}".format(epoch + 1, loss))
