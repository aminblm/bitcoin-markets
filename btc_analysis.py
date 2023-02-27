import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error



def calculate_fundamental_component(df, w):
    """
    Calculates fundamental component using the given dataframe and weights.
    """
    
    df['F_t'] = w[0] + w[1]*df['hash_rate'] + w[2]*df['transaction_volume'] + w[3]*df['mining_difficulty'] + w[4]*df['inflation_rate']
    return df['F_t'], df


def calculate_speculative_component(df, alpha, beta, S_0=0):
    S_t = [S_0]
    len_df = sum(1 for _ in df.iterrows())
    initial_index = 0
    for i, row in df.iterrows():
        initial_index = i
        break
    for i, row in df.iterrows():
        S_t.append(alpha[0]*row.MA + alpha[1]*row.RSI + alpha[2]*row.SO + alpha[3]*row.bitcoin_trend + beta*S_t[i-initial_index-1])
        if i >= initial_index + len_df - 2: break
    
    df['S_t'] = S_t
    
    return df['S_t'], df
    
def calculated_predicted_price(df, F_t, S_t):
    """
    Calculates the predicted price.
    """
    df['P_t'] = F_t + S_t
    return df['P_t'], df


def plot_heterogeneous_agent_model(df):
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the Price, Estimated Price (P), Fundamental Component (F_t), and Speculative Component (S)
    ax.plot(df['time'], df['Price'], label='Price')
    ax.plot(df['time'], df['F_t'], label='Fundamental component')
    ax.plot(df['time'], df['S_t'], label='Speculative component')
    ax.plot(df['time'], df['P_t'], label='Estimated Price')

    # Set the title and labels for the plot
    ax.set_title('Bitcoin Price Components')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    plt.show()
    
    
def plot_price_over_time(df):
    """
    Plots the Bitcoin price over time.
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
    df['year'] = df['time'].dt.year
    plt.plot(df['time'], df['Price'])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Bitcoin Price over Time')
    plt.xticks(rotation=90)
    plt.show()


def plot_hash_rate_vs_mining_difficulty(df):
    """
    Plots the hash rate vs. mining difficulty.
    """
    plt.scatter(df['hash_rate'], df['mining_difficulty'])
    plt.xlabel('Hash Rate')
    plt.ylabel('Mining Difficulty')
    plt.title('Hash Rate vs. Mining Difficulty')
    plt.show()


def plot_transaction_volume_per_year(df):
    """
    Plots the transaction volume per year.
    """
    tv_per_year = df.groupby('year')['transaction_volume'].sum()
    plt.bar(tv_per_year.index, tv_per_year.values)
    plt.xlabel('Year')
    plt.ylabel('Transaction Volume')
    plt.title('Transaction Volume per Year')
    plt.show()


def plot_inflation_rate_over_time(df):
    """
    Plots the inflation rate over time.
    """
    plt.fill_between(df['time'], df['inflation_rate'], color='blue', alpha=0.2)
    plt.legend(['Inflation Rate', 'Interest Rate'])
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.title('Inflation Rate over Time')
    plt.show()


def plot_price_vs_hash_rate_over_time(df):
    """
    Plots the price vs. hash rate over time.
    """
    plt.scatter(df['time'], df['Price'], c=df['hash_rate'], cmap='coolwarm')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('Price vs Hash Rate over Time')
    plt.colorbar(label='Hash Rate')
    plt.show()


def plot_mining_difficulty_by_year(df):
    """
    Plots the mining difficulty by year.
    """
    df_difficulty = df.groupby('year')['mining_difficulty'].sum()
    plt.bar(df_difficulty.index, df_difficulty)
    plt.xlabel('Year')
    plt.ylabel('Mining Difficulty')
    plt.title('Mining Difficulty by Year')
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plots correlation heatmap for the given dataframe.
    """
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Correlation between Variables')
    plt.show()

    
def plot_fundamental_component(df, w):
    """
    Plots fundamental component using the given dataframe and weights.
    """
    F_t, df = calculate_fundamental_component(df, w)
    plt.plot(df['time'], F_t)
    plt.title('Plot of F_t against time')
    plt.xlabel('Time')
    plt.ylabel('F_t')
    plt.show()


def moving_average(df, k):
    ma = df.rolling(window=k).mean()
    return ma


def relative_strength_index(df, k):
    delta = df.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=k).mean()
    avg_loss = loss.rolling(window=k).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def stochastic_oscillator(df, k):
    min_k = df.rolling(window=k).min()
    max_k = df.rolling(window=k).max()
    so = ((df - min_k) / (max_k - min_k)) * 100
    return so


def calculate_technical_sentiment_indicators(df, k):
    df['MA'] = moving_average(df['Price'], k)
    df['RSI'] = relative_strength_index(df['Price'], k)
    df['SO'] = stochastic_oscillator(df['Price'], k)
    
    return df.dropna()
    
def plot_technical_sentiment_indicators(df, k):
    df = calculate_technical_sentiment_indicators(df, k)
    # reset index to 0
    df = df.reset_index(drop=True)
    price = df['Price']
    ma = df['MA']
    rsi = df['RSI']
    so = df['SO']
    gt = df["bitcoin_trend"]

    # Plot indicators
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Bitcoin Indicators', fontsize=16)

    # Moving Average plot
    ax1.plot(df['time'], price, label='Price')
    ax1.plot(df['time'], ma, label='MA')
    ax1.legend()
    ax1.set_ylabel('Price')

    # Relative Strength Index plot
    ax2.plot(df['time'], rsi, label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='r', linestyle='--')
    ax2.legend()
    ax2.set_ylabel('RSI')

    # Stochastic Oscillator plot
    ax3.plot(df['time'], so, label='Stochastic Oscillator')
    ax3.axhline(y=80, color='r', linestyle='--')
    ax3.axhline(y=20, color='r', linestyle='--')
    ax3.legend()
    ax3.set_ylabel('Stochastic Oscillator')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Bitcoin Goole Trend Indicator', fontsize=16)

    # Google Trend plot
    ax1.plot(df['time'], gt, label='GT')
    ax1.legend()
    ax1.set_ylabel('GT Index')

    # Price plot
    ax2.plot(df['time'], df['Price'], label='Price')
    ax2.legend()
    ax2.set_ylabel('Price')

    plt.show()
    
    return df

def calculate_mse_heterogeneous_model(df, weights_fundamental, weights_speculative, beta, S_0):
    F_t, df = calculate_fundamental_component(df, weights_fundamental)
    S_t, df = calculate_speculative_component(df, weights_speculative, beta, S_0)
    df['P_t'], df = calculated_predicted_price(df, F_t, S_t)
    mse = ((df['Price'] - df['P_t'])**2).mean()
    return mse


def optimize_parameters(df, initial_guess_fundamental, initial_guess_speculative, initial_guess_beta, S_0, method='Nelder-Mead'):
    """Here we explain the optimization process in details:
    
    1. We define the objective function, which is the MSE.
    2. We define the initial guess for the weights of the fundamental component, the weights of the speculative component, and the beta.
    3. We use the minimize function from scipy.optimize to minimize the objective function.
    4. scipy.optimize.minimize takes the objective function, the initial guess, and the optimization method as arguments.
    5. minimize uses the Nelder-Mead method by default, but we can also use other methods such as BFGS.
        
    Args:
        df (_type_): _description_
        initial_guess_fundamental (_type_): _description_
        initial_guess_speculative (_type_): _description_
        initial_guess_beta (_type_): _description_
        S_0 (_type_): _description_
        method (str, optional): _description_. Defaults to 'Nelder-Mead'.
    """
    def objective_function(parameters):
        weights_fundamental = parameters[:5]
        weights_speculative = parameters[5:9]
        S_0 = parameters[9]
        beta = parameters[10]
        mse = calculate_mse_heterogeneous_model(df, weights_fundamental, weights_speculative, beta, S_0)
        return mse
    
    initial_guess = np.concatenate([initial_guess_fundamental, initial_guess_speculative, [S_0], [initial_guess_beta]])
    
    result = minimize(objective_function, initial_guess, method=method)
    
    weights_fundamental = result.x[:5]
    weights_speculative = result.x[5:9]
    S_0 = result.x[9]
    beta = result.x[10]
    
    return weights_fundamental, weights_speculative, S_0, beta


def evaluate_model(df):
    mae = mean_absolute_error(df['Price'], df['P_t'])
    mse = mean_squared_error(df['Price'], df['P_t'], squared=True)
    rmse = mean_squared_error(df['Price'], df['P_t'], squared=False)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    
def plot_predicted_prices(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['Price'], label='Actual price')
    plt.plot(df['time'], df['P_t'], label='Predicted price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()