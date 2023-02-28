import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the data
data = btc_df
data = data.dropna()
data = data.drop(['time'], axis=1)

# Preprocess the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Scale data to its original values
data = scaler.

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
test_data = data[train_size:, :]

# Define the VARMA model
def varma_model(data, p, q):
    # Split the data into input and output
    X = data[:, :-1]
    y = data[:, -1]
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    train_X, train_y = X[0:train_size, :], y[0:train_size]
    test_X, test_y = X[train_size:, :], y[train_size:]
    # Fit the VARMA model
    model = VARMAX(train_y, train_X, order=(p, q))
    model_fit = model.fit(maxiter=1000, disp=False)
    # Make predictions on the testing set
    predictions = model_fit.forecast(steps=len(test_y), exog=test_X)
    return predictions

# Define the neural network model
def nn_model(data, n_neurons, n_epochs, n_batch):
    # Split the data into input and output
    X = data[:, :-1]
    y = data[:, -1]
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    train_X, train_y = X[0:train_size, :], y[0:train_size]
    test_X, test_y = X[train_size:, :], y[train_size:]
    # Reshape the input data for the LSTM model
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Fit the LSTM model
    model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    # Make predictions on the testing set
    predictions = model.predict(test_X)
    return predictions.reshape((len(predictions),))

# Define the VARMA-NN model
def varma_nn_model(data, p, q, n_neurons, n_epochs, n_batch, alpha):
    # Fit the VARMA model
    varma_predictions = varma_model(data, p, q)
    # Fit the neural network model
    nn_predictions = nn_model(data, n_neurons, n_epochs, n_batch)
    # Combine the predictions using the weighted average
    predictions = alpha * varma_predictions + (1 - alpha) * nn_predictions
    return predictions

# Define the evaluation metrics
def evaluate_model(test_data, predictions):
    mse = np.mean((test_data - predictions)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    return mse, rmse, mape

# Predict the variables using the VARMA-NN model
p = 1
q = 1
n_neurons = 50
n_epochs = 100
n_batch = 1
alpha = 0.5
predictions = varma_nn_model(data, p, q, n_neurons, n_epochs, n_batch, alpha)
