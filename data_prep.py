import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

import config
days_predict = config.days_predict

def readTrain():
    # train = pd.read_csv("data_set.csv")
    train = pd.read_csv("data_set_validate.csv")
    return train


def extract_feature(df):
    df["log_date"] = pd.to_datetime(df["log_date"])
    df["year"] = df["log_date"].dt.year
    df["month"] = df["log_date"].dt.month
    df["day"] = df["log_date"].dt.day
    df["hour"] = df["log_date"].dt.hour
    df["minute"] = df["log_date"].dt.minute

    df = df.drop(["log_date"], axis=1)
    df = df.drop(["year"], axis=1)
    return df


def normalize(df):
    '''Return numpy'''
    # df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    scaler_minmax = MinMaxScaler()
    data = scaler_minmax.fit_transform(df)
    return data


def split_data(data_np, rate):
    '''Split data, one for training, the other for validation'''
    data_np_1 = data_np[:int(data_np.shape[0] * rate)]
    data_np_2 = data_np[int(data_np.shape[0] * rate):]
    return data_np_1, data_np_2


def build_train(data_np, window=50):
    X_dataset = []
    Y_dataset = []
    X = []
    Y = []
    for i in range(data_np.shape[0]-days_predict):
        X_dataset.append(data_np[i])
        # Add predict data
        Y_dataset.append(data_np[i+days_predict, 0])
    for i in range(len(X_dataset)-window):
        X.append(X_dataset[i:i+window])
        Y.append(Y_dataset[i:i+window])
    return np.array(X), np.array(Y)


train_data_df = readTrain()
train_data_df = extract_feature(train_data_df)
train_data_df_test = copy.deepcopy(train_data_df)
# train_data_df = train_data_df.iloc[:8000]
train_data = normalize(train_data_df)
X_train, Y_train = build_train(train_data, window=50)
Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], -1)
dataset = X_train, Y_train


if __name__ == '__main__':
    print(X_train)
    print(X_train.shape)
    print(Y_train)
    print(Y_train.shape)
    import pickle
    outfile = 'data_feed.pkl'
    with open(outfile,'wb') as f:
        pickle.dump(dataset, f)