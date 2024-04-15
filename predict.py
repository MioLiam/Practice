import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载模型，假设模型名为'mnist_model.h5'，存放在当前目录
model = tf.keras.models.load_model(r'D:\Code\2024.3\python\my_mnist_model.h5')

def predict_digit(img_path):
    # 读取图像
    img = Image.open(img_path).convert('L')  # 转换为灰度图像

    # 调整图像到28x28像素
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 转换图像为numpy数组，并进行归一化
    img_array = np.array(img)
    img_array = img_array / 255.0  # 归一化

    # 显示图像
    plt.imshow(img_array, cmap='gray')
    plt.title("Processed Image")
    plt.show()

    # 重塑数组以符合模型输入要求
    img_array = img_array.reshape(1, 28, 28)

    # 进行预测
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions[0])
    print(f"Predicted digit: {predicted_digit}")

# 使用图像路径调用函数
predict_digit(r'C:\Users\MioLi\Desktop\屏幕截图 2024-04-13 181617.png')