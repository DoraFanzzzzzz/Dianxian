import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 假设你有一个文本文件 `data.txt`，每行是一个文本样本
with open(r"C:\Users\Dora\Desktop\数据集.txt", 'r') as file:
    texts = file.readlines()

# 使用 Tokenizer 将文本转换为数字序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 使用 pad_sequences 进行填充，使所有文本序列长度相同
max_length = 100  # 适当选择合适的序列长度
padded_sequences = pad_sequences(sequences, maxlen=max_length)

def build_generator(latent_dim, num_classes):
    model = keras.Sequential()
    model.add(layers.Embedding(num_classes, 50, input_length=100))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# 定义判别器模型
def build_discriminator(num_classes):
    model = keras.Sequential()
    model.add(layers.Embedding(num_classes, 50, input_length=num_classes))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# 定义生成器和判别器
latent_dim = 100
num_classes = 10
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(num_classes)

# 定义优化器和损失函数
optimizer = keras.optimizers.Adam(learning_rate=0.0002)
loss_fn = keras.losses.BinaryCrossentropy()

# 定义训练循环
num_epochs = 50
batch_size = 128

for epoch in range(num_epochs):
    # 生成随机噪声作为生成器输入
    noise = np.random.randint(0, num_classes, size=(batch_size, latent_dim))
    labels = np.random.randint(0, num_classes, size=(batch_size, num_classes))

    # 使用生成器生成假样本
    fake_samples = generator.predict([noise, labels])
    fake_labels = np.zeros((batch_size, 1))

    # 随机选择真实样本作为训练样本
    real_samples = np.random.randn(batch_size, num_classes)
    real_labels = np.ones((batch_size, 1))

    # 将真实和生成的样本合并在一起
    samples = np.concatenate([real_samples, fake_samples])
    target_labels = np.concatenate([real_labels, fake_labels])

    # 随机打乱样本顺序
    indices = np.random.shuffle(np.arange(2 * batch_size))
    samples = samples[indices]
    target_labels = target_labels[indices]

    # 训练判别器
    with tf.GradientTape() as tape:
        predictions = discriminator(samples)
        d_loss = loss_fn(target_labels, predictions)

    gradients = tape.gradient(d_loss, discriminator.trainable_weights)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))

    # 训练生成器
    with tf.GradientTape() as tape:
        generated_samples = generator([noise, labels])
        predictions = discriminator(generated_samples)
        g_loss = loss_fn(real_labels, predictions)

    gradients = tape.gradient(g_loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(gradients, generator.trainable_weights))

    # 打印每个 epoch 的损失
    print(f"Epoch {epoch + 1}/{num_epochs} - d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")

# 生成生成器生成的样本
noise = np.random.randint(0, num_classes, size=(10, latent_dim))
labels = np.random.randint(0, num_classes, size=(10, num_classes))
generated_samples = generator.predict([noise, labels])

# 打印生成的样本
for i in range(10):
    generated_sample = generated_samples[i]
    print(''.join(str(label) for label in generated_sample))
