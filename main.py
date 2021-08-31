# -*- coding:utf-8 -*-
# Imports
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

# Loading the imdb data (it's preloaded in Keras)
# imdb 数据集来自 IMDB 的 25,000 条电影评论，
# 以情绪（正面/负面）标记。评论已经过预处理，并编码为词索引（整数）的序列表示。
# 为了方便起见，将词按数据集中出现的频率进行索引，
# 例如整数 3 编码数据中第三个最频繁的词。这允许快速筛选操作，
# 例如：「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。
# 作为惯例，0 不代表特定的单词，而是被用于编码任何未知单词。
(x_train, y_train), (x_test, y_test) = imdb.load_data(nb_words=1000)  # load the most 1000 frequent words

print(x_train.shape)
print(x_test.shape)

# Examining the data
print(x_train[0])
print(y_train[0])

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(nb_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])

# One-hot encoding the output
num_classes = 2
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)


# Building the model architecture with one layer of length 1000
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=1000)) # output size 512
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Running and evaluating the model
# 训练网络判去判断（IMDB的25000条电影评论中的）情绪积极还是消极
hist = model.fit(x_train, y_train,
          batch_size=32,
          nb_epoch=10,
          validation_data=(x_test, y_test),
          verbose=1)
