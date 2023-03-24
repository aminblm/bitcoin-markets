# 2017, Narrative Economics, 2005, Speculative Asset Prices, Robert Shiller (Nobel Prize 2013)

Boulouma, A


## Shiller Economic Model for Predicting Bitcoin Prices

The Shiller Economic Model is a widely used tool for predicting the future course of asset prices based on historical data alone. The model is based on the idea that asset prices are influenced not only by fundamental factors, such as economic growth and interest rates, but also by the collective beliefs and emotions of investors.

In his book "Narrative Economics" (2017), Robert Shiller argued that narratives, or stories, play a crucial role in shaping investor beliefs and can have a significant impact on asset prices. In his earlier work "Speculative Asset Prices" (2005), Shiller also highlighted the importance of investor psychology and the role of irrational exuberance in driving asset prices to unsustainable levels.

Using these insights, we can construct a Shiller Economic Model for predicting the future course of bitcoin prices based on historical data alone. The model can be expressed mathematically as follows:

\begin{equation}
P_{t+1} = P_t + \alpha(\beta P_t - \gamma\bar{P}),
\end{equation}

where $P_t$ is the bitcoin price at time $t$, $P_{t+1}$ is the predicted price at time $t+1$, $\bar{P}$ is the average price over a specified period, and $\alpha$, $\beta$, and $\gamma$ are parameters that capture the effects of investor narratives and emotions.

The first term on the right-hand side of the equation ($\beta P_t$) represents the fundamental value of bitcoin, based on factors such as adoption, usage, and network effects. The second term ($-\gamma\bar{P}$) captures the impact of investor narratives and emotions, which can cause prices to deviate from their fundamental value.

The parameter $\beta$ represents the speed of adjustment of bitcoin prices to their fundamental value, while $\gamma$ captures the impact of investor sentiment on price movements. The parameter $\alpha$ determines the weight given to the impact of investor narratives and emotions in the model.

To estimate the parameters of the model, we can use historical data on bitcoin prices and investor sentiment, as captured by measures such as the Google Trends search volume index for "bitcoin". By fitting the model to the data, we can obtain predictions for future bitcoin prices based on past trends in investor beliefs and emotions.

In conclusion, the Shiller Economic Model provides a useful framework for predicting the future course of bitcoin prices based on historical data alone. By accounting for the role of investor narratives and emotions in shaping asset prices, the model can help investors make more informed decisions about their investments in bitcoin and other speculative assets.



# Implementation


```python
from btc_analysis import *
from btc_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
```


```python
def load_data():
    # Load bitcoin prices data
    data = clean_and_transform_data(read_data("../datasets/public/btc.csv"), read_data("../datasets/public/btc_google_trend.csv"))
    prices = data['Price'].values

    # Add P_t_1 column to dataframe
    data['P_t_1'] = np.concatenate([prices[1:], [None]])
    data = data.dropna()

    return data

def split_data(data):
    # Split data into training and testing sets
    split_idx = int(len(data) * 0.8)
    X_train = data['Price'].values[:split_idx]
    X_test = data['Price'].values[split_idx:]
    y_train = data['P_t_1'].values[:split_idx]
    y_test = data['P_t_1'].values[split_idx:]

    return X_train, X_test, y_train, y_test


def compute_p_bar(X_train):
    # Compute P_bar
    P_bar = np.mean(X_train)
    return P_bar


def shiller_model(P, alpha, beta, gamma, P_bar):
    # Define Shiller model function
    return P + alpha * (beta * P - gamma * P_bar)


def loss(params, X_train, y_train, P_bar, lambda_reg):
    # Define loss function with L2 regularization
    alpha, beta, gamma = params
    y_pred = shiller_model(X_train, alpha, beta, gamma, P_bar)
    mse_loss = np.mean((y_train - y_pred)**2)
    l2_reg = lambda_reg * np.sum(params**2)
    return mse_loss + l2_reg


def estimate_optimal_params(X_train, y_train, P_bar):
    # Estimate optimal parameters using training data with regularization
    initial_params = np.array([0.01, 1.0, 0.1]) # initial guess for params
    lambda_reg = 0.1 # regularization strength
    optimal_params = minimize(loss, initial_params, args=(X_train, y_train, P_bar, lambda_reg)).x
    return optimal_params


def evaluate_model(X_test, y_test, alpha, beta, gamma, P_bar):
    # Evaluate model on testing data
    y_pred = shiller_model(X_test, alpha, beta, gamma, P_bar)
    mse = np.mean((y_test - y_pred)**2)
    return mse, y_pred


def fill_shiller(data, alpha, beta, gamma, P_bar):
    # Add predicted values to dataframe
    data.loc[data.index[-len(y_pred):], 'P_t_1_shiller'] = y_pred

    for i in range(1, len(data)-1):
        if np.isnan(data['P_t_1_shiller'][i]):
            data['P_t_1_shiller'][i] = shiller_model(data['Price'][i-1], alpha, beta, gamma, P_bar)

    data['P_t_1_shiller'][0] = data['Price'][1]
    
# Print estimated parameters
def print_params(optimal_param):
    print('Estimated alpha: {:.4f}'.format(optimal_params[0]))
    print('Estimated beta: {:.4f}'.format(optimal_params[1]))
    print('Estimated gamma: {:.4f}'.format(optimal_params[2]))
    
def plot_shiller_predictions(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.time, data['Price'], label='Original Prices')
    plt.plot(data.time, data['P_t_1_shiller'], label='Shiller Model Predictions')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price (USD)')
    plt.show()
    
def residual_plot(data):
    # Plot residuals
    residuals = data['Price'] - data['P_t_1_shiller']
    plt.scatter(data.time, residuals)
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
def plot_future_predictions(data):
    # Generate dates from the end of the available data to the end of 2030
    dates = pd.date_range(data['time'].iloc[-1], '2030-12-31', freq='D')

    # Initialize an empty dataframe to store the predictions
    predictions = pd.DataFrame(index=dates, columns=['P_t_1_shiller'])

    # Set the first prediction to the last available price
    predictions.iloc[0] = data['Price'].iloc[-1]

    # Generate predictions for each date
    for i in range(1, len(predictions)):
        predictions.iloc[i] = predictions.iloc[i-1] + alpha*(beta*predictions.iloc[i-1] - gamma*P_bar)

    # Plot the predictions
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['P_t_1_shiller'], label='Historical data')
    plt.plot(predictions.index, predictions['P_t_1_shiller'], label='Predictions')
    plt.title('Bitcoin price predictions using the Shiller model')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
```


```python
# Load data
data = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(data)

# Compute P_bar
P_bar = compute_p_bar(X_train)

# Estimate optimal parameters using training data with regularization
alpha, beta, gamma = optimal_params = estimate_optimal_params(X_train, y_train, P_bar)

# Evaluate model on testing data
mse, y_pred = evaluate_model(X_test, y_test, alpha, beta, gamma, P_bar)
print('MSE on testing data: {:.4f}'.format(mse))

# Fill in missing values in the "P_t_1_shiller" column of the dataframe using the Shiller model
fill_shiller(data, alpha, beta, gamma, P_bar)

# Plot shiller prediction
plot_shiller_predictions(data)

# Residual plot
residual_plot(data)

# Plot future prices
plot_future_predictions(data)
```
