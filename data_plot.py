import matplotlib.pyplot as plt
import numpy as np
import data_prep
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd

import config
model_name = config.model_name
model_name = 'LSTM_bme280_humi_predict.modelbk1'

# Control
import pickle
outfile = 'data_feed.pkl'
with open(outfile,'rb') as f:
    train_data = pickle.load(f)
X_train, Y_train = train_data
train = Y_train[:, 0, :]

# Predict
model = load_model(model_name)
predict = model.predict(X_train, batch_size=128)


# outfile = 'predit.npy'
# np.save(outfile, predict)

# outfile = 'predit.npy'
# predict = np.load(outfile)



scaler_minmax = MinMaxScaler()
data = scaler_minmax.fit_transform([[x] for x in data_prep.train_data_df['temperature']])

start = 0
batch_size = 800
predict_ori = scaler_minmax.inverse_transform(predict[:, 0][start:start + batch_size])
train_data = scaler_minmax.inverse_transform(train[start:start + batch_size])
# print(train_data)
plt.plot(train_data, label='train_data')
plt.plot(predict_ori, label='predict')

plt.legend()
plt.show()