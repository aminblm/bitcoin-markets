#!/usr/bin/env python
# coding: utf-8

# # Predicting Bitcoin Prices via Mathematical and Financial Models: A Study of the Time-Series Analysis of the economic and macroeconomic factors and of the technical indicators and sentiment analysis
# 
# ## Author: Amin Boulouma

# ### Problem
# 
# The problem is to develop a predictive model for the bitcoin market prices that can accurately forecast the price at time $t+1$ based on the price at time $t$.
# 
# ### Research Question
# 
# What is the best predictive model to use for forecasting bitcoin market prices, and what is the predictive power of each model?
# 
# 

# ## Model 2 - VARMA-NN: A Hybrid Model for Multivariate Time Series Forecasting Using Neural Networks and Vector Autoregressive Moving Average Models
# 
# Time series forecasting is an essential problem in several fields such as finance, economics, and engineering, and has been an active research topic for several decades. Among the existing forecasting models, the VARMA-NN model is a more powerful and flexible extension of the ARIMA-NN model, as it can handle multivariate time series data and capture both linear and non-linear dependencies between variables. The VARMA-NN model combines the VARMA model with a neural network model, similar to the ARIMA-NN model. This paper discusses the VARMA-NN model proposed by Zhang, G. P. (2003) in detail and its implementation to forecast economic and macroeconomic indicators, technical indicators, and sentiment analysis for predicting future values.
# 
# The VARMA model is a generalization of the VAR model, where it includes the moving average terms in addition to the autoregressive terms. The VARMA model is specified by two parameters: the order of the autoregressive terms $p$ and the order of the moving average terms $q$. The VARMA model can be written as:
# 
# $$\sum_{i=1}^{p} \Phi_i L^i \Delta Y_t = \sum_{j=1}^{q} \Theta_j L^j \epsilon_t$$
# 
# where $\Delta Y_t = (Y_t - Y_{t-1})$ is the differenced time series, $L$ is the lag operator, $\Phi_i$ and $\Theta_j$ are the autoregressive and moving average coefficient matrices, respectively, and $\epsilon_t$ is the white noise error term at time $t$.
# 
# The VARMA-NN model combines the VARMA model with a neural network model in a similar way to the ARIMA-NN model. The VARMA-NN model can be formalized as follows:
# 
# 1. Preprocessing: The time series data is preprocessed to remove any outliers or missing values. The data is then split into training and testing sets.
# 
# 2. VARMA modeling: The VARMA model is fit to the training data. The VARMA model is specified by two parameters: $p$ and $q$. The VARMA model captures the linear dependencies between past and future values of the multivariate time series.
# 
# 3. Neural network modeling: A neural network model is built using the training data. The neural network can be a feedforward neural network or a recurrent neural network (RNN). The neural network captures the non-linear dependencies that may exist in the data.
# 
# 4. Hybrid modeling: The predictions from the VARMA model and the neural network model are combined using a weighted average. The weights are determined by the relative performance of the two models on the training data.
# 
# 5. Evaluation: The performance of the VARMA-NN model is evaluated using the testing data. The evaluation metrics used can include mean squared error (MSE), root mean squared error (RMSE), and mean absolute percentage error (MAPE).
# 
# 6. Interpretation: Interpret the model results and use them to inform decision-making. It is important to keep in mind that the model results are not a crystal ball and should be used in conjunction with other factors and expert judgment.
# 
# 7. Updating the model: As new data becomes available, the model should be updated and refined to ensure it remains accurate and relevant.
# 
# The equations used in the VARMA-NN model can be represented as:
# 
# ### VARMA model:
# 
# $$\sum_{i=1}^{p} \Phi_i L^i \Delta Y_t = \sum_{j=1}^{q} \Theta_j L^j \epsilon_t$$
# 
# where $\Delta Y_t$ is the differenced multivariate time series, $\Phi_i$ and $\Theta_j$ are the autoregressive and moving average coefficient matrices, respectively, and $\epsilon_t$ is the white noise error term at time $t$.
# 
# ### Neural network model:
# 
# $$Y_t = f(WX_t + b)$$
# 
# where $Y_t$ is the predicted value at time $t$, $X_t$ is the input vector at time $t$, $W$ is the weight matrix, $b$ is the bias vector, and $f$ is the activation function.
# 
# ### Hybrid model:
# 
# $$Y_t = \alpha Y_t^{VARMA} + (1 - \alpha) Y_t^{NN}$$
# 
# where $Y_t^{ARIMA}$ is the prediction from the ARIMA model, $Y_t^{NN}$ is the prediction from the neural network model, and $\alpha$ is the weight assigned to the ARIMA prediction.
# 
# The specific formulas for each indicator are as follows:
# 
# The moving average indicator ($MA$): The formula for $MA$ with a window size of $k$ can be written as:
# $$Y_{MA,t} = \frac{1}{k} \sum_{i=t-k+1}^{t} X_i$$
# 
# where $X_i$ is the value of the variable at time $i$.
# 
# The relative strength index ($RSI$): The formula for $RSI$ with a window size of $k$ can be written as:
# $$Y_{RSI,t} = 100 - \frac{100}{1 + RS}$$
# 
# where $RS$ is the relative strength at time $t$, which is calculated as:
# 
# $$RS = \frac{\sum_{i=t-k+1}^{t} Max(X_i - X_{i-1}, 0)}{\sum_{i=t-k+1}^{t} |X_i - X_{i-1}|}$$
# 
# The stochastic oscillator ($SO$): The formula for $SO$ with a window size of $k$ can be written as:
# $$Y_{SO,t} = \frac{X_t - Min_{k}(X)}{Max_{k}(X) - Min_{k}(X)} \times 100$$
# 
# where $Min_{k}(X)$ and $Max_{k}(X)$ are the minimum and maximum values of the variable over the past $k$ periods, respectively.
# 
# The Google Trend indicator $f_{GT}(Q_t)$: The formula for $f_{GT}(Q_t)$ is:
# $$Y_{GT,t} = f_{GT}(Q_t)$$
# 
# where $Q_t$ represents the search query related to Bitcoin at time $t$, and $f_{GT}$ is a function that processes the search data to generate a Google Trends score.
# 
# Overall, the VARMA-NN model offers a powerful and flexible approach to time series forecasting, particularly for multivariate data with both linear and non-linear dependencies. It combines the strengths of both the VARMA and neural network models, allowing it to capture complex relationships between variables.
# 
# However, it is important to note that the model is not a one-size-fits-all solution and must be tailored to the specific data and problem at hand. It also requires a significant amount of data and computational resources to train and optimize, so it may not be suitable for all applications.
# 
# Nonetheless, the VARMA-NN model represents a significant advancement in time series forecasting and has the potential to greatly improve our ability to predict future trends and outcomes. Further research and development in this area are likely to yield even more powerful and effective forecasting methods in the future.
# 
# ### Long Short-Term Memory (LSTM) network
# 
# The model defined in the provided code is based on a recurrent neural network (RNN), specifically a Long Short-Term Memory (LSTM) network. The LSTM network is used to capture the temporal dependencies and patterns in the time series data.
# 
# Given a time series data of length $T$, where $y_t$ denotes the observation at time $t$, the LSTM network takes a sequence of $n$ time steps as input and predicts the value of the next observation in the sequence. In this implementation, the number of time steps $n$ is defined as n_steps.
# 
# The input data is first normalized using the MinMaxScaler function from scikit-learn. This scales each feature to the range of 0 to 1, which helps to improve the training performance of the LSTM network.
# 
# The LSTM network is then defined using the Keras library. The network consists of a single LSTM layer with 50 units, followed by a dense output layer with the same number of features as the input data. The network is trained using the mean squared error (MSE) loss and the Adam optimizer.
# 
# To make predictions for a specific variable at a given time, the model takes the previous $n$ observations of all the variables as input and outputs the predicted value for the variable of interest. The predict_variable function takes as input the name of the variable to predict (variable) and the time for which the prediction is requested (time). The function retrieves the previous $n$ observations of all the variables up to the given time and scales the data using the same MinMaxScaler used during training. The scaled input data is then reshaped to match the expected input shape of the LSTM network and fed into the network to obtain the predicted value. The predicted value is then inverse transformed using the MinMaxScaler to obtain the predicted value in the original scale.
# 
# The formal mathematical representation of the LSTM model used in the implementation is as follows:
# 
# Given a sequence of $n$ time steps, $\boldsymbol{X}={x_1, x_2, \dots, x_n}$, where $x_t \in \mathbb{R}^{p}$ denotes the $p$-dimensional input at time $t$, the output of an LSTM network can be represented as:
# 
# $$\begin{aligned} \boldsymbol{h}_t &= \text{LSTM}(\boldsymbol{x}t, \boldsymbol{h}{t-1}), \ \boldsymbol{y}_t &= \boldsymbol{W}_o \boldsymbol{h}_t + \boldsymbol{b}_o, \end{aligned}$$
# 
# where $\boldsymbol{h}_t \in \mathbb{R}^{m}$ denotes the hidden state at time $t$, $\boldsymbol{W}_o \in \mathbb{R}^{p \times m}$ and $\boldsymbol{b}_o \in \mathbb{R}^{p}$ denote the output weights and biases, respectively.
# 
# During training, the network parameters are learned by minimizing the mean squared error (MSE) loss between the predicted output and the actual output:
# 
# $$\text{MSE} = \frac{1}{T-n}\sum_{t=n+1}^T (\boldsymbol{y}_t - \boldsymbol{\hat{y}}_t)^2,$$
# 
# where $\boldsymbol{y}_t$ denotes the true output at time $t$ and $\boldsymbol{\hat{y}}_t$ denotes the predicted output at time $t$.
# 
# The Negative Mean Squared Error is a performance metric commonly used to evaluate the quality of regression models. It is defined as the negative of the mean squared error, which measures the average of the squared differences between the predicted and actual values of the target variable. Mathematically, the neg_mean_squared_error can be expressed as:
# 
# $$\text{NMSE} = -\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
# 
# where $n$ is the number of samples, $y_i$ is the actual value of the target variable for the $i$th sample, and $\hat{y}_i$ is the predicted value of the target variable for the $i$th sample.
# 
# The negative sign in the definition of the NMSE is used to ensure that higher values of the metric correspond to better performance, i.e., lower mean squared error. This is because many optimization algorithms are designed to minimize the objective function, so minimizing the negative of the mean squared error is equivalent to maximizing the mean squared error.
# 
# Overall, the NMSE provides a measure of how well the regression model fits the data and can be used to compare the performance of different regression models.
# 
# **References**:
# 
# - Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.
# 
# - Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159-175.
# 
# - Box, G. E. P., & Jenkins, G. M. (1976). Time series analysis: Forecasting and control. San Francisco, CA: Holden-Day.
# 
# - Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and practice. OTexts.
# 
# - Chollet, F. (2018). Deep learning with Python. Manning Publications.
# 
# - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.

# ## 1. Predict economic indicators

# ### 1.1. LTSM

# In[435]:


from btc_analysis import *
from btc_data import *

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.varmax import VARMAX
import math
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressorimport pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[72]:


btc_df = clean_and_transform_data(read_data("datasets/btc.csv"), read_data("datasets/btc_google_trend.csv"))


# In[118]:


plot_correlation_heatmap(btc_df)


# > This implementation uses a hybrid ARIMA and neural network model to predict all the indicators in the given dataset. The ARIMA model is used to generate short-term forecasts, while the neural network model is used to generate long-term forecasts. The two models are then combined to generate a final forecast.

# > We nee a model that trains a VARMAX model on the given time series data and allows you to make predictions for any variable of your choice based on the time input:
# 
# 

# > One possible way to improve the model architecture could be to use a deep learning model, such as a recurrent neural network (RNN), to capture the temporal dependencies and patterns in the time series data.
# 
# 

# This script loads the time series data from a CSV file, scales it using MinMaxScaler, trains a VARMAX model with order (1,1), and defines a function predict_variable that takes a variable name and a time input and returns the predicted value for that variable at that time. The predict_variable function uses the trained model to make predictions by preparing an input dataframe with the given time input and filling any missing values with forward and backward filling. It then scales the input data and uses the forecast method of the trained model to make a prediction for the scaled input data. The predicted value is then inverse scaled and returned as the final output.

# In[ ]:


# > We need a model that trains a VARMAX model on the given time series data and allows you to make predictions for any variable of your choice based on the time input:
# > One possible way to improve the model architecture could be to use a deep learning model, such as a recurrent neural network (RNN), to capture the temporal dependencies and patterns in the time series data.


# Load the data
df = btc_df = clean_and_transform_data(read_data("datasets/btc.csv"), read_data("datasets/btc_google_trend.csv"))

# Set the date column as the index
df.set_index('time', inplace=True)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Define the number of time steps for the input data
n_steps = 3

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size,:]
test_data = scaled_data[train_size-n_steps:,:]

# Define the input and output data for the model
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i,:])
        y.append(data[i,:])
    X, y = np.array(X), np.array(y)
    return X, y

train_X, train_y = prepare_data(train_data, n_steps)
test_X, test_y = prepare_data(test_data, n_steps)

# Define the Keras model
def create_model():
    # Define the RNN model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, df.shape[1])))
    model.add(Dense(df.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

regressor = KerasRegressor(build_fn=create_model)

# Define the hyperparameters to be tuned
param_grid = {
    'batch_size': [16, 32],
    'epochs': [50, 100],
}

# Define the GridSearchCV object
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, verbose=1)

# Perform the GridSearchCV
grid_result = grid_search.fit(train_X, train_y)

# Print the best parameters and the corresponding mean squared error
print("Best parameters: ", grid_result.best_params_)
print("Best MSE: ", grid_result.best_score_)


# In[ ]:


# Train the model using the best hyperparameters
best_model = Sequential()
best_model.add(LSTM(units=6, activation='relu', input_shape=(n_steps, df.shape[1])))
best_model.compile(optimizer='adam', loss='mse')
best_model.fit(train_X, train_y, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'], verbose=0)

# # Train the model
# model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)


# In[524]:


n_steps = 3

def predict_variable(variable, time):
    # Check if the time is in the dataframe
    if time in df.index:
        # Use the available data to make the prediction
        input_times = pd.date_range(start=time-np.timedelta64(n_steps-1, 'D'), end=time, freq='D')
    else:
        closest_time = df.index[df.index.get_loc(time, method='nearest')]
        # Use the available data to make the prediction
        input_times = pd.date_range(start=closest_time-np.timedelta64(n_steps-1, 'D'), end=closest_time, freq='D')
        
    # Prepare the input data
    input_data = df.loc[input_times,:]
    input_data_scaled = scaler.transform(input_data)
    input_data_reshaped = input_data_scaled.reshape(1, n_steps, df.shape[1])
    
    # Make predictions for the variable
    predictions_scaled = best_model.predict(input_data_reshaped).flatten()
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, df.shape[1]))
    
    return predictions[0][df.columns.get_loc(variable)]


# In[549]:


def build_dataset(model, time):
    # can I use a trained model to build a dataset based on time input alone?
    # Yes, you can use a trained model to build a dataset based on time input alone, especially if the model was trained on time series data. One common technique for generating new data based on time input is to use the model to forecast future values of the time series.
    # For example, if you have a model that was trained to predict the price of a stock based on historical price data, you could use the model to generate a dataset of future price predictions by providing the model with a sequence of future time steps.
    # However, it's important to note that the quality of the generated data will be highly dependent on the accuracy of the model and the assumptions made about the future behavior of the time series. Additionally, generating new data based on time input alone may not capture all the relevant features and factors that contribute to the time series behavior, so it's important to carefully consider the limitations and assumptions of the approach.
    # Build dataset from model only
    df = 
    return df

    


# In[536]:


# Example usage:
predicted_price = predict_variable("Price", pd.Timestamp("2023-01-01"))
print(f"Predicted BTC price on 2021-01-01: {predicted_price:.2f}")


# In[550]:


# Example usage:
predicted_price = predict_variable_without_df("Price", pd.Timestamp("2023-01-01"))
print(f"Predicted BTC price on 2021-01-01: {predicted_price:.2f}")


# In[405]:


# Evaluate the model
def evaluate_model():
    for varname in df.columns:
        
        # Make predictions for the testing set
        predictions_scaled = model.predict(test_X)
        predictions = scaler.inverse_transform(predictions_scaled)
        
        test_data_df = pd.DataFrame(test_data, columns=df.columns)
        predictions_df = pd.DataFrame(predictions, columns=df.columns)
        
        mse = ((predictions_df[varname] - test_data_df[varname]) ** 2).mean()
        mae = abs(predictions_df[varname] - test_data_df[varname]).mean()
        
        print(f"Root Mean Squared Error for {varname}: {math.sqrt(mse):.2f}")
        print(f"Mean Absolute Error for {varname}: {mae:.2f}")


# We use a recurrent neural network (RNN) to train a VARMAX model on the given time series data and makes predictions for any variable based on the time input. 

# In[485]:


# Example usage:
predicted_price = predict_variable("Price", pd.Timestamp("2025-01-01"))
print(f"Predicted BTC price on 2021-01-01: {predicted_price:.2f}")


# In[478]:


# Example usage:
predicted_price = predict_variable("Price", pd.Timestamp("2015-01-01"))
print(f"Predicted BTC price on 2028-01-01: {predicted_price:.2f}")


# In[406]:


evaluate_model()


# In[366]:


import matplotlib.pyplot as plt

def plot_variable_vs_predicted(variable, time):
    # Get the actual values of the variable for the specified period of time
    actual_data = df.loc[time:, variable]
    
    # Get the predicted values of the variable for the same period of time
    predicted_data = []
    for t in actual_data.index:
        predicted_data.append(predict_variable(variable, t))
    predicted_data = pd.Series(predicted_data, index=actual_data.index)
    
    # Plot the actual and predicted values
    plt.figure(figsize=(12,6))
    plt.plot(actual_data.index, actual_data, label='Actual')
    plt.plot(predicted_data.index, predicted_data, label='Predicted')
    plt.title(f"{variable} vs Predicted {variable}")
    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.legend()
    plt.show()


# In[537]:


plot_variable_vs_predicted("Price", "2022-01-01")


# In[ ]:


plot_variable_vs_predicted("Price", "2022-01-01")


# In[ ]:


def plot_predicted(variable, start_date, end_date):
    # Get the predicted values of the variable for the specified period of time
    predicted_data = []
    for t in pd.date_range(start=start_date, end=end_date):
        predicted_data.append(predict_variable(variable, t))
    predicted_data = pd.Series(predicted_data, index=pd.date_range(start=start_date, end=end_date, freq='D'))

    # Plot the predicted values
    plt.figure(figsize=(12,6))
    plt.plot(predicted_data.index, predicted_data, label='Predicted')
    plt.title(f"Predicted {variable}")
    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.legend()
    plt.show()


# In[ ]:


import pickle

# Save
with open('models/best_ltsm_model.pkl', 'wb') as file:
    pickle.dump(grid_result.best_estimator_, file)


# In[ ]:


# Load the model from a file
with open('models/best_ltsm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[370]:


for col in df.columns:
    plot_variable_vs_predicted(col, pd.Timestamp("2022-01-01"))


# 

# ### 1.2. VARMAX

# In[429]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = btc_df = clean_and_transform_data(read_data("datasets/btc.csv"), read_data("datasets/btc_google_trend.csv"))

# Set the date column as the index
df.set_index('time', inplace=True)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Define the number of time steps for the input data
n_steps = 3

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size,:]
test_data = scaled_data[train_size-n_steps:]

# Define a function to prepare the input and output data for the model
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i,:])
        y.append(data[i,:])
    X, y = np.array(X), np.array(y)
    return X, y

train_X, train_y = prepare_data(train_data, n_steps)
test_X, test_y = prepare_data(test_data, n_steps)

# Train the VARMAX model
model = VARMAX(endog=train_y, exog=train_X, order=(1,1))
result = model.fit()

# Define a function to make predictions for any variable
def predict_variable(variable, time):
    # Prepare the input data
    input_data = df.loc[time-np.timedelta64(n_steps-1, 'D'):time,:]
    input_data_scaled = scaler.transform(input_data)
    input_data_reshaped = input_data_scaled.reshape(1, n_steps, train_X.shape[2])
    input_data_reshaped = input_data_reshaped[:, :, :-1]

    # Make predictions for the variable
    predictions_scaled = result.forecast(exog=input_data_reshaped, steps=n_steps)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    return predictions[0][df.columns.get_loc(variable)]

# Example usage:
predicted_price = predict_variable("Price", pd.Timestamp("2022-01-01"))
print(f"Predicted BTC price on 2022-01-01: {predicted_price:.2f}")

