from PIL import Image
import numpy as np
from tensorflow.keras.datasets import mnist

# 加载外部图像
external_image = Image.open(r'C:\Users\MioLi\Desktop\屏幕截图 2024-04-13 181617.png')

# 转换为灰度图像
external_image = external_image.convert('L')

# 调整图像大小到28x28
external_image = external_image.resize((28, 28), Image.Resampling.LANCZOS)

# 将图像转换为NumPy数组
external_image_array = np.array(external_image)

# 归一化像素值到0-1
external_image_array = external_image_array.astype('float32') / 255.0

# 如果需要查看图像，可以使用以下代码
import matplotlib.pyplot as plt
plt.imshow(external_image_array, cmap='gray')
plt.show()

external_label = np.array([7])



# 加载原始的MNIST数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 假设external_image_array是你已经预处理的外部图像数组
# 假设external_label是你的外部图像的标签

# 添加外部图像到训练集图像数组
train_images = np.concatenate((train_images, external_image_array[np.newaxis, :, :]), axis=0)

# 添加外部标签到训练集标签数组
train_labels = np.concatenate((train_labels, external_label), axis=0)