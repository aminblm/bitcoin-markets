#!/usr/bin/env python
# coding: utf-8

# # Predicting Bitcoin Prices via Mathematical and Financial Models: A Study of the Time-Series Analysis of the seasonal, trend, and residual components, A comprehensive use of STL Model
# 
# > Amin

# Seasonal decomposition of time series (STL) is a statistical technique used to decompose a time series into its seasonal, trend, and residual components. STL is particularly useful for analyzing time series data with non-linear and non-stationary trends, and for identifying seasonal patterns in the data.
# 
# The STL algorithm works by first smoothing the time series with a moving average to remove the high-frequency noise. The smoothed series is then decomposed into its seasonal, trend, and residual components using a process called loess regression. The seasonal component represents the periodic pattern in the data, such as weekly or monthly fluctuations. The trend component represents the long-term trend in the data, while the residual component represents the random fluctuations or noise that cannot be explained by the seasonal or trend components.
# 
# The STL algorithm can be represented mathematically as follows:
# 
# Let y be a time series of length n, and let S, T, and R be the seasonal, trend, and residual components, respectively. The STL algorithm can be formulated as:
# 
# Smooth the time series y with a moving average of window length m to obtain a smoothed series s.
# Compute the seasonal component S by subtracting the seasonal subseries from the smoothed series s. The seasonal subseries is obtained by averaging the values of y at each seasonal position (e.g., for monthly data, the seasonal subseries for January is the average of all January values in the data set).
# Compute the trend component T by applying loess regression to the detrended series. The detrended series is obtained by subtracting the seasonal component S from the smoothed series s.
# Compute the residual component R by subtracting the seasonal component S and the trend component T from the original series y.
# The seasonal, trend, and residual components can be combined to obtain a reconstructed time series that closely approximates the original time series.
# 
# Given a time series $y_t$ for $t=1,2,...,T$, the STL method decomposes it into three components: trend $T_t$, seasonal $S_t$ and remainder $R_t$ as follows:
# 
# Loess Smoothing:
# The first step in STL is to extract the trend component by applying a loess smoother to the original time series. Let $m$ be the degree of the polynomial used in the loess smoother. The smoothed values, $\hat{T}_t$, are given by:
# 
# $$ \hat{T}t = \ell_t + \sum{j=1}^m b_j L_j(t) $$
# 
# where $\ell_t$ is the local polynomial regression estimate of the trend at time $t$, $b_j$ are the smoothing parameters, and $L_j(t)$ are the $j$th degree Legendre polynomials evaluated at time $t$.
# 
# Detrending:
# The detrended values, $y^*_t$, are obtained by subtracting the smoothed values from the original time series:
# 
# $$ y^*_t = y_t - \hat{T}_t $$
# 
# Seasonal Smoothing:
# The seasonal component is extracted by applying a seasonal smoother to the detrended values. Let $s$ be the length of the seasonal period, and $q$ be the degree of the polynomial used in the seasonal smoother. The seasonal values, $\hat{S}_t$, are given by:
# 
# $$ \hat{S}t = \frac{1}{K} \sum{j=1}^K y^*_{t + (j-1)s} $$
# 
# where $K = \lfloor \frac{T}{s} \rfloor$ is the number of seasonal periods in the time series.
# 
# Deseasonalizing:
# The deseasonalized values, $y^{**}_t$, are obtained by dividing the detrended values by the seasonal values:
# 
# $$ y^{**}_t = \frac{y^*_t}{\hat{S}_t} $$
# 
# Residual Smoothing:
# The remainder component is obtained by applying a smoother to the deseasonalized values. Let $r$ be the degree of the polynomial used in the residual smoother. The smoothed residuals, $\hat{R}_t$, are given by:
# 
# $$ \hat{R}t = \sum{j=1}^r c_j \epsilon_{t-j} $$
# 
# where $c_j$ are the smoothing parameters, and $\epsilon_{t-j}$ are the residuals at time $t-j$.
# 
# Reconstruction:
# The final step in STL is to add the trend, seasonal and remainder components back together to obtain the reconstructed values, $\hat{y}_t$:
# 
# $$ \hat{y}_t = \hat{T}_t + \hat{S}_t \cdot y^{**}_t + \hat{R}_t $$
# 
# **Strengths**:
# 
# - STL is a well-established time series decomposition method that has been widely used in various applications.
# - STL can effectively handle time series data with different types of seasonality, trend, and noise components.
# - The method is flexible and allows for the adjustment of parameters such as the length of the seasonal window and the degree of smoothing.
# - The decomposition results can provide insights into the underlying patterns of the time series, which can be useful in forecasting and anomaly detection.
# - STL can handle missing values and outliers in the time series data.
# 
# **Weaknesses**:
# 
# - STL is computationally expensive, especially for long and high-frequency time series data.
# - The method may not work well for non-stationary time series data with complex patterns, such as abrupt changes in trend or seasonality.
# - The decomposition results may be sensitive to the choice of parameters, such as the degree of smoothing and the seasonal window length.
# - The method assumes that the seasonal pattern is fixed and deterministic, which may not be true for some time series data.
# 
# **Opportunities**:
# 
# - The popularity of time series analysis in various industries and applications creates opportunities for further development and improvement of STL.
# - The increasing availability of computing resources can help to overcome the computational challenges of STL and make it more accessible for large-scale time series data analysis.
# - The use of STL in combination with other time series analysis techniques, such as machine learning models, can enhance the forecasting accuracy and predictive power of the method.
# 
# **Threats**:
# 
# - The development of new time series decomposition methods that are more computationally efficient and can handle more complex time series patterns may reduce the competitiveness of STL.
# - The increasing availability of machine learning models for time series analysis may reduce the demand for traditional time series decomposition methods like STL.
# - The potential changes in the underlying patterns of the time series data due to external factors, such as economic downturns or global pandemics, may affect the accuracy and reliability of the STL decomposition results.
# 
# ## References
# - STL has been introduced by R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning in their paper "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", published in the Journal of Official Statistics, Vol. 6, No. 1, 1990.
# - Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice. OTexts.

# ## Implementation

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint

from btc_analysis import *
from btc_data import *

# Load the data
df = btc_df = clean_and_transform_data(read_data("datasets/btc.csv"), read_data("datasets/btc_google_trend.csv"))

# Set the date column as the index
df.set_index('time', inplace=True)

# df.columns
# Index(['Price', 'hash_rate', 'transaction_volume', 'mining_difficulty','inflation_rate', 'bitcoin_trend'], dtype='object')

# Build the STL Model
def stl_model(df, freq):
    """ Build the STL model for time series decomposition

    Args:
        df (dataframe): the dataframe containing the time series data
        freq (int): the frequency of the time series data, examples: 7 for weekly data, 12 for monthly data
    """
    # Decompose the time series into trend, seasonal, and residual components
    # decomposition = STL(df, period=freq).fit()
    decomposition = STL(df.squeeze(), period=freq).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the trend, seasonal, and residual components
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    trend.plot(ax=axes[0])
    axes[0].set_title('Trend')
    seasonal.plot(ax=axes[1])
    axes[1].set_title('Seasonal')
    residual.plot(ax=axes[2])
    axes[2].set_title('Residual')
    plt.show()

    # Plot the original and reconstructed time series
    df_reconstructed = trend + seasonal + residual
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    df.plot(ax=axes[0])
    axes[0].set_title('Original')
    df_reconstructed.plot(ax=axes[1])
    axes[1].set_title('Reconstructed')
    plt.show()
    
    fig = decomposition.plot()

    return decomposition


# Calibrate the model

def calibrate_stl_model(df, freq):
    """ Calibrate the STL model by iterating through different seasonal window lengths and polynomial degrees

    Args:
        df (dataframe): the dataframe containing the time series data
        freq (int): the frequency of the time series data, examples: 7 for weekly data, 12 for monthly data, 365 daily
    """
    min_rmse = 10000000000
    best_model = STL(df)
    
    # Iterate through different seasonal window lengths and polynomial degrees
    for window_length in range(3, 10, 2):
        for degree in range(1, 5):
            # Split data on train and test
            train_size = int(len(df) * 0.8)
            train, test = df[0:train_size], df[train_size:len(df)]
            
            # Build the STL model
            decomposition = STL(train, period=freq, seasonal=window_length, robust=True, seasonal_deg=degree).fit()
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Plot the original and reconstructed time series
            df_reconstructed = trend + seasonal + residual
            # apply the model to the test data
            # df_reconstructed = df_reconstructed.append(decomposition.predict(len(test)))
            # use the forcast method for tofecast the price based on the time series
            df_reconstructed = df_reconstructed.append(decomposition.forecast(test.shape[0]))
            # AttributeError: 'DecomposeResult' object has no attribute 'predict'
            print(test, df_reconstructed[train_size:len(df)])
            # Calculate the RMSE from the test and train
            rmse = sqrt(mean_squared_error(test, df_reconstructed[train_size:len(df)]))
            
            print('RMSE: %.3f' % rmse)

            print('Window Length: %d, Degree: %d' % (window_length, degree))
            print('-------------------------------------')
            
            if rmse < min_rmse:
                min_rmse = rmse
                best_model = decomposition
    return best_model

# Test the model

def 

# Plot the results

# ## Conclusion

