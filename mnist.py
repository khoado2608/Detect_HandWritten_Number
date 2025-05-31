import numpy as np
from tensorflow.keras.datasets import mnist

# Tải dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Lưu dataset vào thư mục hiện tại
np.savez("mnist_dataset.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print("MNIST dataset đã được lưu vào mnist_dataset.npz trong thư mục hiện tại.")
