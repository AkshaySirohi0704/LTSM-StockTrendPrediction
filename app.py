import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import keras.models as load_model
import streamlit as st
import tensorflow as tf

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock trend prediction')
user_input = st.text_input('Enter Stock Tcker', 'AAPL')
print('Started ... ')

# Describing Data
#df = data.DataReader(user_input, 'yahoo')
df = data.DataReader(user_input, 'yahoo')
print(df.head())
print(df.tail())
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualiation
st.subheader('CLOSING PRICE VS TIME CHART')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Time Graph vs Time Chart with 100DMA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart 100DMA and 200DMA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Spliting Data into Training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print('Spliting Data into Training and Testing ... ')

# print(data_testing.head())

# DEEP LEARNING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dTraingArray = scaler.fit_transform(data_training) # scaling data between 0 and 1

#modelCheck = load_model("C:\Flying Raijin\StockTrendPrediction(Using LTSM Model\stockPredictionLTSM(F).h5")
modelCheck = tf.keras.models.load_model("stockPredictionLTSM(F).h5")

# Testing Data
past100Days = data_training.tail(100)
#finalDf = past100Days.append(data_testing, ignore_index=True)
finalDf = pd.concat([past100Days,data_testing], ignore_index=True)
inputData = scaler.fit_transform(finalDf) # Scalling testing data
print('Scalling data ... ')
#print("Input Data Shape : ",inputData.shape)
x_test = []
y_test = []

for i in range(100, inputData.shape[0]):
    x_test.append(inputData[i-100:i])
    y_test.append(inputData[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test) # coverting array to numpy

print('Predicting ... ')
y_predicted = modelCheck.predict(x_test)
y_predicted = np.array(y_predicted)
scaler = scaler.scale_

scalerFactor = 1/scaler[0]
y_predicted = y_predicted*scalerFactor
y_test = y_test*scalerFactor

# Droping unecessory Colomns
#dfRecent = data.DataReader(user_input, 'yahoo')
#print(dfRecent.tail())
#dfRecent = dfRecent.reset_index()



st.subheader('Predicted price Vs Orginal price')
fig = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'b', label = "Predicted Price")
plt.plot(y_test, 'r', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

'''
st.subheader('Todays Predictions')
y_pred = np.reshape(y_predicted, len(y_predicted))
print(y_pred[len(y_pred)-2:len(y_pred)])
y_pred = y_pred[len(y_pred)-2:len(y_pred)]
print(y_test[len(y_test)-2:len(y_test)])
y_temp = y_test[len(y_test)-2:len(y_test)]

#printDf = pd.DataFrame({"yesterday":[y_pred[0],y_temp[0]], "Today":[y_pred[1],y_temp[1]]}, columns=['Predicted', 'Original'])
# Inject CSS with Markdown
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)
printDf = pd.DataFrame({"Day":['Yesterday','Today'], "Predicted":[y_pred[0],y_pred[1]], "Real":[y_temp[0],y_temp[1]], "Per Day Offset":[y_test[0]-y_pred[0],y_test[1]-y_pred[1]]})
print('Lenght of y_Prediction : ', len(y_predicted))
print('Lenght of y_Test : ', len(y_test))
st.table(printDf)
'''







