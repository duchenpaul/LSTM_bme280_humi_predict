import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import config

model_name = config.model_name
batch_size = 128


def buildModel(shape):
    model = Sequential()
    model.add(
        LSTM(6, input_shape=(shape[1], shape[2]), return_sequences=True, batch_size=batch_size))
    model.add(Dropout(0.1))
    # output shape: (1, 1)
    # model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    # split training data and validation data
    import pickle
    outfile = 'data_feed.pkl'
    with open(outfile,'rb') as f:
        train_data = pickle.load(f)

    X_train, Y_train = train_data
    model = buildModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

    model.fit(X_train, Y_train, epochs=1000, shuffle=True, batch_size=batch_size, validation_split=0.1, callbacks=[callback, tbCallBack])
    model.save(model_name)
