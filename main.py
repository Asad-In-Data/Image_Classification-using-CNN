import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
from tensorflow.keras import layers, models,datasets

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# normalize pixel values to be between 0 and 1
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

class_names = ['airplane', 'automobile', 'bird', 
               'cat', 'deer', 'dog', 'frog',
                 'horse', 'ship', 'truck']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.xlabel("Class: " + class_names[train_labels[i][0]])
    plt.axis("off")
plt.show()
