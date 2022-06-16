from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from tensorflow import keras
#import keras.models as model
from tensorflow.python.keras.models import load_model

start = '2010-01-01'
end = '2019-12-31'

# Discovering Data
df = data.DataReader('AAPl', 'yahoo')
# print(df.head())
# print(df.tail())

df = df.reset_index()
# Droping unecessory Colomns
df = df.drop(['Date', 'Adj Close', 'Volume'], axis=1)
# print(df.head())

# moving average of 100 days and 200 days
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Visualizing
plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
#plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
#plt.show()

# Spliting Data into Training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# print(data_testing.head())

# DEEP LEARNING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dTraingArray = scaler.fit_transform(data_training) # scaling data between 0 and 1

x_train = []
y_train = []

for i in range(100, data_training.shape[0]):
    x_train.append(dTraingArray[i-100:i])
    y_train.append(dTraingArray[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) # coverting array to numpy

# dl model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
# Layer 1
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
# Layer 2
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
# Layer 3
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
# Layer 4
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
# Layer 5
model.add(Dense(units=1))
#model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 50)
model.save("C:\Flying Raijin\StockTrendPrediction(Using LTSM Model)\stockPredictionLTSM(F).h5")

# Testing Data
past100Days = data_training.tail(100)
finalDf = past100Days.append(data_testing, ignore_index=True)
inputData = scaler.fit_transform(finalDf) # Scalling testing data

x_test = []
y_test = []

for i in range(100, inputData.shape[0]):
    x_test.append(inputData[i-100:i])
    y_test.append(inputData[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test) # coverting array to numpy

# Prediting 
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scaleFactor = 1/0.02099517
y_predicted = y_predicted*scaleFactor
y_test = y_test*scaleFactor



# Visualizing
plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label = 'Predicted Price')
#plt.plot(y_test, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
plt.show()
