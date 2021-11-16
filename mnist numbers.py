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



"""
when its run
1500/1500 [==============================] - 5s 2ms/step - loss: 0.2879 - accuracy: 0.9179 - val_loss: 0.1655 - val_accuracy: 0.9499
Epoch 2/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1286 - accuracy: 0.9624 - val_loss: 0.1142 - val_accuracy: 0.9661
Epoch 3/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0878 - accuracy: 0.9741 - val_loss: 0.0988 - val_accuracy: 0.9702
Epoch 4/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0652 - accuracy: 0.9806 - val_loss: 0.0933 - val_accuracy: 0.9733
Epoch 5/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0487 - accuracy: 0.9862 - val_loss: 0.0874 - val_accuracy: 0.9751
Epoch 6/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0390 - accuracy: 0.9881 - val_loss: 0.0869 - val_accuracy: 0.9750
Epoch 7/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0300 - accuracy: 0.9911 - val_loss: 0.0869 - val_accuracy: 0.9757
Epoch 8/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0239 - accuracy: 0.9928 - val_loss: 0.0889 - val_accuracy: 0.9763
Epoch 9/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0201 - accuracy: 0.9940 - val_loss: 0.0894 - val_accuracy: 0.9772
Epoch 10/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0154 - accuracy: 0.9954 - val_loss: 0.0946 - val_accuracy: 0.9768
Epoch 11/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0117 - accuracy: 0.9968 - val_loss: 0.1001 - val_accuracy: 0.9746
Epoch 12/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0127 - accuracy: 0.9959 - val_loss: 0.0920 - val_accuracy: 0.9783
Epoch 13/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0086 - accuracy: 0.9975 - val_loss: 0.1020 - val_accuracy: 0.9767
Epoch 14/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0093 - accuracy: 0.9974 - val_loss: 0.1063 - val_accuracy: 0.9771
Epoch 15/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0070 - accuracy: 0.9980 - val_loss: 0.1005 - val_accuracy: 0.9786
Epoch 16/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0062 - accuracy: 0.9980 - val_loss: 0.1147 - val_accuracy: 0.9756
Epoch 17/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0070 - accuracy: 0.9979 - val_loss: 0.1148 - val_accuracy: 0.9772
Epoch 18/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0049 - accuracy: 0.9987 - val_loss: 0.1199 - val_accuracy: 0.9771
Epoch 19/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0061 - accuracy: 0.9981 - val_loss: 0.1395 - val_accuracy: 0.9717
Epoch 20/20
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0052 - accuracy: 0.9985 - val_loss: 0.1218 - val_accuracy: 0.9780
313/313 [==============================] - 1s 2ms/step - loss: 0.1108 - accuracy: 0.9772
Accuracy: 0.9771999716758728, Loss: 0.11076284199953079 """
