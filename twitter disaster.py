import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from keras.layers import LeakyReLU, PReLU


nltk.download("stopwords")

df = pd.read_csv(r"C:\Users\xavie\OneDrive\Desktop\IRP\data\twitter_disasters\train.csv")

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_Punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

pattern = re.compile(r"https?://\S+|www\.\S+")
for t in df.text:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break

df["text"] = df.text.map(remove_URL)
df["text"] = df.text.map(remove_Punct)

stop = set(stopwords.words("english"))

def remove_Stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

df["text"] = df.text.map(remove_Stopwords)

def counter_Word(text_col):
    count = Counter()

    for text in text_col.values:
        for word in text.split():
            count[word] += 1

    return count

counter = counter_Word(df.text)

num_unique_words = len(counter)
#print(counter.most_common(5))

train_size = int(df.shape[0] * 0.8)


train_df = df[:train_size]
val_df = df[train_size:]

test_size = int(train_df.shape[0] *0.8)

test_df = train_df[test_size:]

train_df = train_df[:test_size]

train_sentences = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()

val_sentences = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()

test_sentences = test_df.text.to_numpy()
test_labels = test_df.target.to_numpy()
#print(train_sentences.shape, val_sentences.shape)

tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

max_length = 20

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

reverse_word_index = dict([(idx, word) for word, idx in word_index.items()])

def decode(sequence):
    return ' '.join([reverse_word_index.get(idx, "?") for idx in sequence])

model = keras.models.Sequential()

model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

#leakyRelu = LeakyReLU()
#paraRelu = PReLU()

model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, ))
model.summary()

loss = keras.losses.BinaryCrossentropy(from_logits = False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)
model.fit(train_padded, train_labels, epochs=20, validation_data = (val_padded, val_labels), verbose = 2)


test_loss, test_acc = model.evaluate(test_padded, test_labels)


print("Accuracy: {}, Loss: {}".format(test_acc, test_loss))
