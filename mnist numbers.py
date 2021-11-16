import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU

(x_train, x_labels), (y_train, y_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

classnames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(type(x_train))

'''
print(x_labels[2])
plt.imshow(x_train[2], cmap=plt.cm.binary)
plt.show()
'''

x_train = x_train / 255.0
y_train = y_train / 255.0

total_len = len(x_train)
val_data = x_train[48000:]
val_labels = x_labels[48000:]
print(len(val_data))
print(type(val_data))
x_train = x_train[:48000]
x_labels = x_labels[:48000]
print(len(x_train))
print(type(x_train))

#leakyRelu = LeakyReLU(alpha=0.01)# if below 0 then it is multiplied by alpha
#pararelu = PReLU() #its a leakyrelu but the alpha is affected by backpropagation

model = keras.Sequential(layers=
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),# change this for different activation functions
        keras.layers.Dense(10, activation="softmax")
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, x_labels, epochs=20, validation_data=(val_data, val_labels))

test_loss, test_acc = model.evaluate(y_train, y_labels)

print("Accuracy: {}, Loss: {}".format(test_acc, test_loss))

