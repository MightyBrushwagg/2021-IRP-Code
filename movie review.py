import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import LeakyReLU, PReLU

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88_000) # only use the 88,000 most common words


word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0 # is added onto the end of lists that are shorter than the required length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding="post", maxlen=256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# model here

#leakyRelu = LeakyReLU(alpha=0.01)
#pararelu = PReLU()

model = keras.Sequential()
model.add(keras.layers.Embedding(88_000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, ))
model.add(keras.layers.Dense(1, ))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")

x_val = train_data[:10_000]
x_train = train_data[10_000:]

y_val = train_labels[:10_000]
y_train = train_labels[10_000:]
print("training: {}".format(len(x_train)))
print("validatiing: {}".format((len(x_val))))
print("testing: {}".format((len(test_labels))))

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data = (x_val, y_val), verbose=1)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print("Accuracy: {}, Loss: {}".format(test_acc, test_loss))
