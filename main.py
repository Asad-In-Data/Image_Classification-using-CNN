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

train_images=train_images[:20000]
train_labels=train_labels[:20000]
test_images=test_images[:40000]   
test_labels=test_labels[:40000]

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.xlabel("Class: " + class_names[train_labels[i][0]])
    plt.axis("off")
plt.show()

# As model is created now we just load it ... for creating uncomment this
# Create the model
# model=models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images,train_labels,epochs=10,batch_size=64)
# model.evaluate(test_images,test_labels)

# model.save('My_model.keras') 

model=models.load_model('My_model.keras')
