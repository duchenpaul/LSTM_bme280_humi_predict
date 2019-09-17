import matplotlib.pyplot as plt
import numpy as np
import data_prep
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

import config
model_name = config.model_name
records_predict = config.records_predict

# Control
import pickle
outfile = 'data_feed.pkl'
with open(outfile,'rb') as f:
    train_data = pickle.load(f)
X_train, Y_train = train_data
# train = Y_train[:, 0, :]
train = Y_train[:, 0, :]


# Predict
model = load_model(model_name)
predict = model.predict(X_train, batch_size=128)[records_predict:, :, :]
print(X_train.shape)
print(predict.shape)
# outfile = 'predit.npy'
# np.save(outfile, predict)

# outfile = 'predit.npy'
# predict = np.load(outfile)



scaler_minmax = MinMaxScaler()
data = scaler_minmax.fit_transform([[x] for x in data_prep.train_data_df['temperature']])

start = 0
batch_size = 383
predict_ori = scaler_minmax.inverse_transform(predict[:, 0][start:start + batch_size])
train_data = scaler_minmax.inverse_transform(train[start:start + batch_size])
rms = sqrt(mean_squared_error(train_data, predict_ori))
print('RMSE: {}'.format(rms))

# print(train_data)
plt.plot(train_data, label='train_data')
plt.plot(predict_ori, label='predict')
plt.plot(train_data-predict_ori, label='error')
plt.grid(True, linestyle = "-.")

plt.legend()
plt.show()

