from operator import imul
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import PIL
from keras.layers import LeakyReLU, PReLU

IMAGE_SIZE = (100, 100)

#dataset - https://www.kaggle.com/vijaykumar1799/face-mask-detection

# 0 - mask right, 1 - mask wrong, 2 - no mask
class_names = ["Mask worn correctly", "mask worn incorrectly", "no mask worn"]

datagen = keras.preprocessing.image.ImageDataGenerator()

directory = r"C:\Users\xavie\OneDrive\Desktop\IRP\data\mask_wearing_data"

train_it = datagen.flow_from_directory(os.path.join(directory, "train"), class_mode="binary", batch_size=64, shuffle=True, target_size=IMAGE_SIZE, color_mode="grayscale")
val_it = datagen.flow_from_directory(os.path.join(directory, "validation"), class_mode="binary", batch_size=64, shuffle=True, target_size=IMAGE_SIZE, color_mode="grayscale")
test_it = datagen.flow_from_directory(os.path.join(directory, "test"), class_mode="binary", batch_size=64, shuffle=True, target_size=IMAGE_SIZE, color_mode="grayscale")

batch_X, batch_y = train_it.next()

#print("Batch shape: {}, batch min: {}, batch max: {}".format(batch_X.shape, batch_X.min(), batch_X.max()))

#leakyRelu = LeakyReLU(0.01)
#paraRelu = PReLU()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100,100,1)),
    keras.layers.Dense(1000, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_it,  validation_data=val_it, epochs=20)


test_loss, test_acc = model.evaluate(test_it)

print("Accuracy: {}, Loss: {}".format(test_acc, test_loss))
