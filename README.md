# Bitcoin Markets

This project is a collection of Jupyter notebooks and scientific papers that analyze Bitcoin markets using economical, financial and machine learning models.

## Getting Started

### Prerequisites

First, you'd need to be able to write and understand Python code, I have written a book coined by one of its readers as the Bible of Python: [Python Programming: A Comprehensive Guide to Software Development with Real-World Applications](https://www.amazon.com/Python-Programming-Comprehensive-Development-Application/dp/B0BW2G3W2R/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr=)

You can purchase this book on Amazon to support its author and the author of this work in this repository.

### Overview

This project is a collection of Jupyter notebooks that analyze Bitcoin markets using economical, financial and machine learning models.

You can find the full list of notebooks and models in here:

- [Integrating Interdisciplinary Models for Precise Forecasting of Bitcoin Market Prices: A Fusion of Financial, Economic, Time-Series, and Machine and Deep Learning Approaches](notebooks/Combining%20Financial,%20Economic,%20Time-Series,%20Mathematical%20and%20Deep%20Learning%20Models%20for%20Accurate%20Forecasting%20of%20Bitcoin%20Market%20Prices.ipynb)

- [Combining Economic Models for Bitcoin Market Forecasting: A Comparative Analysis](notebooks/Economic%20Models%20for%20Bitcoin%20Maket%20Forcast.ipynb)

We also develop a separate implementation and a separate notebook for each model in this project.

Our goal is to provide those models, datasets and analysis for the community.

## Models

## Beauty Contest model of Keynes, 1936

The Beauty Contest model was introduced by John Maynard Keynes in his 1936 book "The General Theory of Employment, Interest, and Money." The model describes a situation in which participants make decisions based not on their own opinions, but on their perception of what other participants are likely to do. In the context of financial markets, this can lead to the creation of speculative bubbles.

To apply the Beauty Contest model to Bitcoin prices, we can simulate a group of participants and ask them to guess the future price of Bitcoin. Each participant's guess will be based on their perception of what other participants are likely to guess.

The mathematical equation for the Beauty Contest model is as follows:

$$P_n = \frac{1}{n}\sum_{i=1}^{n}w_i\cdot f_i$$

where,

$P_n$ is the average guess of the $n$ participants
$w_i$ is the weight assigned to participant $i$
$f_i$ is the guess of participant $i$

In the case of the Beauty Contest model of Keynes, the weights are assigned based on the perceived influence of each participant on the market. This can be modeled using a power law distribution, where the weight assigned to participant $i$ is proportional to their perceived influence:

$$w_i = k \cdot i^{-\alpha}$$

where,

$k$ is a normalization constant
$\alpha$ is the power law exponent, which controls the distribution of weights

By adjusting the value of $\alpha$, we can simulate different scenarios in which the market is influenced more or less by the opinions of a few highly influential participants.

> **This model may be useful for predicting market sentiment and how it may affect Bitcoin prices.**

### Model training data, input data, output data, integrations
- Data: Historical price data and trading volume of bitcoin from cryptocurrency exchanges, social media sentiment data
- Input: The winning prediction is the average of predictions made by multiple individuals, where each person tries to guess the outcome of the market based on the actions of other individuals rather than the underlying fundamentals.
- Output: The predicted price of bitcoin in the short-term based on the consensus of the crowd.
- Combining models: This model can be combined with sentiment analysis of social media data to predict the expected behavior of market participants.

## Artificial Neural Networks (ANN): The concept of artificial neural networks has its roots in the work of Warren McCulloch and Walter Pitts in the 1940s. The development of modern neural networks, including backpropagation, can be attributed to multiple researchers including Paul Werbos in 1974, David Rumelhart and James McClelland in 1986, and Geoffrey Hinton in the 2000s

Artificial neural networks are a type of machine learning algorithm that are modeled after the structure of the human brain. An ANN consists of layers of interconnected nodes, with each node performing a simple mathematical operation. The equations for a feedforward ANN with one hidden layer are:

$$z_j = \sum_{i=1}^n w_{ij} x_i + b_j$$

$$h_j = \sigma(z_j)$$

$$y_k = \sum_{j=1}^m v_{jk} h_j + c_k$$

where $x_i$ is the input to the network, $w_{ij}$ is the weight between the $i$-th input and $j$-th hidden layer node, $b_j$ is the bias of the $j$-th hidden layer node, $z_j$ is the weighted sum of the inputs to the $j$-th hidden layer node, $\sigma(\cdot)$ is the activation function (e.g. sigmoid, ReLU), $h_j$ is the output of the $j$-th hidden layer node, $v_{jk}$ is the weight between the $j$-th hidden layer node and the $k$-th output node, $c_k$ is the bias of the $k$-th output node, and $y_k$ is the output of the network.

> **ANN can be useful for predicting Bitcoin prices based on a large set of input data that has complex patterns and nonlinear relationships.**

### Model training data, input data, output data, integrations
- Data: Historical price data and trading volume of bitcoin from cryptocurrency exchanges
- I Input: The historical prices of bitcoin, and other relevant factors such as trading volume, and volatility measures.
- Output: The predicted price of bitcoin in the short, mid, and long-term based on historical data and identified patterns.
- Combining models: ANN can be combined with GARCH and LSTM models to make better long-term forecasts.

## Bayesian regression model: Bayesian regression has been developed by multiple researchers including Harold Jeffreys in the 1940s and Bruno de Finetti in the 1950s

Bayesian regression is a statistical model that allows for prior knowledge about the parameters to be incorporated into the model. The model can be written as:

\begin{equation}
y_i = \boldsymbol{x}_i^{\top} \boldsymbol{\beta} + \epsilon_i, \qquad i = 1, \dots, n,
\end{equation}

where $y_i$ is the dependent variable, $\boldsymbol{x}_i$ is a vector of independent variables, $\boldsymbol{\beta}$ is a vector of regression coefficients, and $\epsilon_i$ is a random error term. In Bayesian regression, the prior distribution for $\boldsymbol{\beta}$ is specified as:

\begin{equation}
\boldsymbol{\beta} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\end{equation}

where $\mathcal{N}$ denotes the normal distribution, $\boldsymbol{\mu}$ is a vector of prior means, and $\boldsymbol{\Sigma}$ is a prior covariance matrix.

> **Bayesian regression can be useful for predicting Bitcoin prices when there is a limited amount of data available or when the data has a high level of noise or uncertainty.**

### Model training data, input data, output data, integrations
- Data: Historical price data of bitcoin and other cryptocurrencies
- Input: The historical prices of bitcoin, and other relevant factors such as trading volume, and volatility measures.
- Output: The predicted price of bitcoin in the short, mid, and long-term based on historical data and probability estimates.
- Combining models: Bayesian regression can be combined with VAR and ARIMA models to make better short-term and long-term forecasts.

## Vector autoregression (VAR) model: Introduced by Clive Granger in the 1960s

The VAR model is a time series model that captures the dynamics of multiple time series variables simultaneously. The VAR model can be written as:

$$y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + \epsilon_t$$

where $y_t$ is a $K \times 1$ vector of variables and $\epsilon_t$ is a $K \times 1$ vector of error terms at time $t$. The $A_i$ matrices are $K \times K$ coefficient matrices for each lag, with $p$ representing the number of lags included in the model. The VAR model assumes that each variable in the system is a function of its own lagged values as well as the lagged values of the other variables in the system.

The VAR model can be estimated using least squares or maximum likelihood methods. Once the model is estimated, it can be used for forecasting by iteratively predicting the values of the variables in the system using their own past values and the past values of the other variables in the system.

> **VAR can be useful for predicting the interrelationships between different variables that may affect Bitcoin prices, such as exchange rates, interest rates, and other economic indicators.**

### Model training data, input data, output data, integrations
- Data: Historical price data and trading volume of bitcoin from cryptocurrency exchanges
- Input: The historical prices of bitcoin, and other relevant factors such as trading volume, and volatility measures.
- Output: The predicted price of bitcoin in the short and mid-term based on past data and interrelated variables.
- Combining models: VAR can be combined with GARCH and ARIMA models to make better short and long-term forecasts.

## ARIMA model: Autoregressive Integrated Moving Average (ARIMA) models: Developed by George Box and Gwilym Jenkins in the 1970s

The ARIMA model is used for time series analysis and forecasting. It is represented as ARIMA(p,d,q), where p is the order of the autoregressive (AR) part, d is the order of differencing, and q is the order of the moving average (MA) part. The equation for an ARIMA model is:

$$\phi_p(B)(1-B)^d X_t = \theta_q(B) \epsilon_t$$

where $X_t$ is the time series, $\epsilon_t$ is the error term, $\phi_p$ and $\theta_q$ are the AR and MA polynomials of order p and q, respectively, $B$ is the backshift operator, and $(1-B)^d$ represents the differencing operation.

> **ARIMA can be useful for predicting Bitcoin prices based on trends, seasonality, and previous price patterns.**

### Model training data, input data, output data, integrations
- Data: Historical price data of bitcoin and other cryptocurrencies
- Input: The historical prices of bitcoin, and other relevant factors such as trading volume, and volatility measures.
- Output: The predicted price of bitcoin in the short and mid-term based on past data and trends.
- Combining models: ARIMA can be combined with VAR and GARCH models to make better mid and long-term forecasts.

## GARCH model: Developed by Robert Engle in the 1980s

The GARCH model is a time series model that captures the dynamics of conditional volatility, where the variance of the error term at time $t$ is a function of the errors at previous times. The GARCH model is specified as follows:

$$y_t = \mu_t + \epsilon_t$$

$$\epsilon_t = \sigma_t z_t$$

$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2$$

where $y_t$ is the time series at time $t$, $\mu_t$ is the conditional mean of the time series, $\epsilon_t$ is the error term at time $t$, $z_t$ is a random variable with a standard normal distribution, $\sigma_t^2$ is the conditional variance of the error term at time $t$, $\omega$ is a constant, $\alpha_i$ and $\beta_j$ are coefficients that determine the weight of the squared error terms and the past variances in the current variance, and $p$ and $q$ are the orders of the autoregressive and moving average terms, respectively.

> **GARCH can be useful for predicting Bitcoin prices when there is volatility clustering and conditional heteroscedasticity in the data.**

### Model training data, input data, output data, integrations

- Data: Historical price data and trading volume of bitcoin from cryptocurrency exchanges
- Input: The historical prices of bitcoin, and other relevant factors such as trading volume, and volatility measures.
- Output: The predicted volatility of bitcoin in the short and mid-term based on past data.
- Combining models: GARCH can be combined with VAR and ARIMA models to make better short and long-term forecasts.

## Markov regime-switching model: Developed by Andrew Harvey in the 1980s

The Markov regime-switching model is a statistical model that allows for changes in the parameters of a time series model based on an unobserved Markov process. It is typically used to model data with non-constant volatility or mean, and is especially useful for financial time series analysis.

The basic model assumes that the observed data is generated by a switching process that follows a discrete-time Markov chain. At each point in time, the model switches between different regimes with different distributions, and the transition probabilities between regimes depend only on the current state of the Markov chain.

Let $y_t$ denote the observed data at time $t$, and let $s_t$ denote the unobserved state of the Markov chain at time $t$. The Markov regime-switching model can be expressed as:

$$
\begin{aligned}
y_t &= \mu_{s_t} + \epsilon_t \\
\epsilon_t &\sim N(0, \sigma^2_{s_t}) \\
s_t &\sim \text{Discrete}(\boldsymbol{\pi}_{s_{t-1}})
\end{aligned}
$$

where $\mu_{s_t}$ and $\sigma^2_{s_t}$ are the mean and variance parameters of regime $s_t$, $\boldsymbol{\pi}_{s_{t-1}}$ is the probability distribution of transitioning from regime $s_{t-1}$ to $s_t$, and $\epsilon_t$ is an error term that is normally distributed with mean zero and variance $\sigma^2_{s_t}$. The discrete distribution indicates that $s_t$ takes on a finite number of possible values.

> **Markov regime-switching can be useful for predicting changes in market regimes and how they may affect Bitcoin prices.**

### Model training data, input data, output data, integrations
- Data: Time series data with seasonal patterns. Two sources of such data are FRED (Federal Reserve Economic Data) and Yahoo Finance.
- Best input: The time series data with seasonal patterns.
- Best output: Decomposition of the time series into its seasonal, trend, and remainder components.
- Models that can be combined: ARIMA, VAR, and GARCH models.

## Support vector regression (SVR) model: Developed by Vladimir Vapnik and Alexey Chervonenkis in the 1990s

The support vector regression (SVR) model is a type of supervised learning algorithm used in machine learning for regression analysis. It was developed by Vladimir Vapnik and Alexey Chervonenkis in the 1990s.

The basic idea of SVR is to map the input data to a higher-dimensional feature space using a kernel function, and then find a linear regression function that maximizes the margin between the predicted output and the actual output. The margin is defined as the distance between the hyperplane that separates the predicted outputs and the actual outputs, and the closest points to the hyperplane.

Let $\mathbf{x}$ denote the input vector, and let $y$ denote the output value. The SVR model can be expressed as:

$$
\begin{aligned}
y &= \mathbf{w}^T\Phi(\mathbf{x}) + b \\
\text{subject to } & y_i - \mathbf{w}^T\Phi(\mathbf{x}_i) - b \le \epsilon \\
& \mathbf{w}^T\Phi(\mathbf{x}_i) + b - y_i \le \epsilon \\
& \mathbf{w}^T\mathbf{w} \le C
\end{aligned}
$$

where $\Phi(\mathbf{x})$ is the mapping function that maps the input vector $\mathbf{x}$ to a higher-dimensional feature space, $\mathbf{w}$ and $b$ are the weight vector and bias term of the linear regression function, $\epsilon$ is the tolerance parameter that controls the size of the margin, and $C$ is the regularization parameter that controls the trade-off between maximizing the margin and minimizing the error. The constraints ensure that the predicted outputs are within a certain distance $\epsilon$ from the actual outputs, and that the weight vector $\mathbf{w}$ has a norm smaller than or equal to $C$.

> **SVR can be useful for predicting Bitcoin prices based on a limited number of input variables, where a nonlinear relationship may exist between the input and output variables.**

### Model training data, input data, output data, integrations

- Data: Time series data with historical prices and related variables such as trading volume, market capitalization, and sentiment data. Two sources of such data are CoinMarketCap and CryptoCompare.
- Best input: Historical prices and related variables as input features.
- Best output: Prediction of future prices.
- Models that can be combined: LSTM, Random Forest, and VAR models.

## The Log-Periodic Power Law (LPPL) by Didier Sornette, 1990

The Log-Periodic Power Law (LPPL) model is a tool used to predict speculative bubbles in financial markets. The LPPL model is based on the theory that asset prices can experience exponential growth rates in the short term, but will eventually experience a crash as market participants realize that the asset is overvalued.

The LPPL model is defined by the following equation:

$$\ln(P_t) = A + B(t_{c-t})^\beta + C(t_{c-t})^\beta\cos[\omega\ln(t_{c-t}) - \phi]$$

where,

$P_t$ is the price of the asset at time $t$
$t_c$ is the critical time of the bubble
$A, B, C, \beta, \omega,$ and $\phi$ are the parameters of the model.

To apply the LPPL model to Bitcoin prices, we first need to gather historical price data for Bitcoin. We can do this by accessing an API that provides historical price data, such as the Coinbase API.

Once we have the historical price data, we can fit the LPPL model to the data using nonlinear regression. The LPPL model has several parameters that need to be estimated, including the critical time, the amplitude, and the frequency.

After estimating the LPPL parameters, we can use the model to predict when a speculative bubble is likely to occur. Specifically, we can look for signs of a divergence between the predicted price and the actual price, which is an indication that a bubble may be forming.

> **LPPL can be useful for predicting when a bubble may be forming in the Bitcoin market and when it may be likely to burst.**

### Model training data, input data, output data, integrations
- Data: Time series data with patterns of large bubbles and crashes. Two sources of such data are CoinDesk and BitMEX.
- Best input: Historical price data of the asset.
- Best output: Prediction of the timing and magnitude of an upcoming crash.
- Models that can be combined: SVR, LSTM, and VAR models.

## The Seasonal decomposition of time series (STL) by R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning (1990)

Seasonal decomposition of time series (STL) is a statistical technique used to decompose a time series into its seasonal, trend, and residual components. STL is particularly useful for analyzing time series data with non-linear and non-stationary trends, and for identifying seasonal patterns in the data.

The STL algorithm works by first smoothing the time series with a moving average to remove the high-frequency noise. The smoothed series is then decomposed into its seasonal, trend, and residual components using a process called loess regression. The seasonal component represents the periodic pattern in the data, such as weekly or monthly fluctuations. The trend component represents the long-term trend in the data, while the residual component represents the random fluctuations or noise that cannot be explained by the seasonal or trend components.

The STL algorithm can be represented mathematically as follows:

Let y be a time series of length n, and let S, T, and R be the seasonal, trend, and residual components, respectively. The STL algorithm can be formulated as:

Smooth the time series y with a moving average of window length m to obtain a smoothed series s.
Compute the seasonal component S by subtracting the seasonal subseries from the smoothed series s. The seasonal subseries is obtained by averaging the values of y at each seasonal position (e.g., for monthly data, the seasonal subseries for January is the average of all January values in the data set).
Compute the trend component T by applying loess regression to the detrended series. The detrended series is obtained by subtracting the seasonal component S from the smoothed series s.
Compute the residual component R by subtracting the seasonal component S and the trend component T from the original series y.

The seasonal, trend, and residual components can be combined to obtain a reconstructed time series that closely approximates the original time series.

Given a time series $y_t$ for $t=1,2,...,T$, the STL method decomposes it into three components: trend $T_t$, seasonal $S_t$ and remainder $R_t$ as follows:

- **Loess Smoothing**:
The first step in STL is to extract the trend component by applying a loess smoother to the original time series. Let $m$ be the degree of the polynomial used in the loess smoother. The smoothed values, $\hat{T}_t$, are given by:

$$ \hat{T}t = \ell_t + \sum{j=1}^m b_j L_j(t) $$

where $\ell_t$ is the local polynomial regression estimate of the trend at time $t$, $b_j$ are the smoothing parameters, and $L_j(t)$ are the $j$th degree Legendre polynomials evaluated at time $t$.

- **Detrending**:

The detrended values, $y^*_t$, are obtained by subtracting the smoothed values from the original time series:

$$ y^*_t = y_t - \hat{T}_t $$

- **Seasonal Smoothing**:

The seasonal component is extracted by applying a seasonal smoother to the detrended values. Let $s$ be the length of the seasonal period, and $q$ be the degree of the polynomial used in the seasonal smoother. The seasonal values, $\hat{S}_t$, are given by:

$$ \hat{S}t = \frac{1}{K} \sum{j=1}^K y^*_{t + (j-1)s} $$

where $K = \lfloor \frac{T}{s} \rfloor$ is the number of seasonal periods in the time series.

- **Deseasonalizing**:

The deseasonalized values, $y^{**}_t$, are obtained by dividing the detrended values by the seasonal values:

$$ y^{**}_t = \frac{y^*_t}{\hat{S}_t} $$

- **Residual Smoothing**:

The remainder component is obtained by applying a smoother to the deseasonalized values. Let $r$ be the degree of the polynomial used in the residual smoother. The smoothed residuals, $\hat{R}_t$, are given by:

$$ \hat{R}t = \sum{j=1}^r c_j \epsilon_{t-j} $$

where $c_j$ are the smoothing parameters, and $\epsilon_{t-j}$ are the residuals at time $t-j$.

- **Reconstruction**:

The final step in STL is to add the trend, seasonal and remainder components back together to obtain the reconstructed values, $\hat{y}_t$:

$$ \hat{y}_t = \hat{T}_t + \hat{S}_t \cdot y^{**}_t + \hat{R}_t $$

> **STL can be useful for predicting Bitcoin prices when there are seasonal patterns or trends in the data.**

### Model training data, input data, output data, integrations

- Data: Time series data with seasonal patterns. Two sources of such data are FRED (Federal Reserve Economic Data) and Yahoo Finance.
- Best input: The time series data with seasonal patterns.
- Best output: Decomposition of the time series into its seasonal, trend, and remainder components.
- Models that can be combined: ARIMA, VAR, and GARCH models.

## Multivariate GARCH (MGARCH) models: Developed by Robert Engle in the 1990s

Developed by Robert Engle in the 1990s, the MGARCH model is an extension of the univariate GARCH model to multivariate time series. It models the conditional variance of each series as a function of its own lagged values and the lagged values of the other series in the system. The mathematical equation for a MGARCH(p, q) model is as follows:

$$\boldsymbol{\Sigma}_{t} = \boldsymbol{A}_{0} + \sum_{i=1}^{p} \boldsymbol{A}_{i} \boldsymbol{\epsilon}_{t-i} \boldsymbol{\epsilon}_{t-i}^{\prime} \boldsymbol{A}_{i}^{\prime} + \sum_{j=1}^{q} \boldsymbol{B}_{j} \boldsymbol{\Sigma}_{t-j} \boldsymbol{B}_{j}^{\prime}$$

where,

- $\boldsymbol{\Sigma}_{t}$ is the $k \times k$ covariance matrix of the error terms at time $t$
- $\boldsymbol{A}_{0}$ is the $k \times k$ intercept matrix
- $\boldsymbol{A}_{i}$ and $\boldsymbol{B}_{j}$ are $k \times k$ coefficient matrices for the error term and covariance matrix, respectively.
- $p$ and $q$ are the order of autoregression and moving average terms for the error term and covariance matrix, respectively.

> **MGARCH can be useful for predicting the interrelationships between different variables that may affect Bitcoin prices, including the volatility of different cryptocurrencies.**

### Model training data, input data, output data, integrations

- Data: Multivariate time series data with volatility clustering and spillover effects. Two sources of such data are - FRED and Quandl.
- Best input: Historical prices of multiple assets, along with other relevant variables.
- Best output: Prediction of future volatility of each asset, and their correlations.
- Models that can be combined: VAR, LSTM, and BNN models.

## Bayesian Neural Networks (BNN): Bayesian neural networks have been developed by multiple researchers including David MacKay in the 1990s and Radford Neal in 1995

The Bayesian neural network model is a type of neural network that incorporates Bayesian inference in the training process. It works by placing a prior distribution over the model parameters and using Bayes' rule to update the posterior distribution of the parameters given the training data. The model uses Monte Carlo methods, such as Markov chain Monte Carlo (MCMC), to approximate the posterior distribution.

The mathematical equation for the Bayesian neural network model is as follows:

Given a training set ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$, where $x_i$ is the input data and $y_i$ is the corresponding output, the goal is to find a function $f(x)$ that predicts the output $y$ for a given input $x$.

The Bayesian neural network model places a prior distribution $p(w)$ over the model parameters $w$ and uses Bayes' rule to update the posterior distribution $p(w|D)$ of the parameters given the training data $D$:

$$p(w|D) = \frac{p(D|w) p(w)}{p(D)}$$

where $p(D|w)$ is the likelihood of the data given the parameters, $p(w)$ is the prior distribution, and $p(D)$ is the marginal likelihood of the data.

> **BNN can be useful for predicting Bitcoin prices based on a large set of input data, where the relationships between the input and output variables may be complex and nonlinear.**

### Model training data, input data, output data, integrations
- Data: Time series data with complex nonlinear relationships. Two sources of such data are Kaggle and GitHub.
- Best input: Historical prices and related variables as input features.
- Best output: Prediction of future prices and their uncertainty.
- Models that can be combined: SVR, LSTM, and Random Forest models.

## Long Short-Term Memory (LSTM) models: Developed by Sepp Hochreiter and Jürgen Schmidhuber in 1997

LSTM models are a type of recurrent neural network that are designed to capture long-term dependencies in time series data. The equations for an LSTM cell are:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

$$\tilde{C}t = \tanh(W_C x_t + U_C h{t-1} + b_C)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

where $x_t$ is the input at time $t$, $h_{t-1}$ is the previous hidden state, $W_f, U_f, b_f$ are the weights and bias for the forget gate, $W_i, U_i, b_i$ are the weights and bias for the input gate, $W_C, U_C, b_C$ are the weights and bias for the cell state, $W_o, U_o, b_o$ are the weights and bias for the output gate, $\sigma(\cdot)$ is the sigmoid activation function.

> **LSTM can be useful for predicting Bitcoin prices based on a large set of input data, where the relationships between the input and output variables may be complex and nonlinear, and where there may be time lags in the data.**

### Model training data, input data, output data, integrations
- Data: Time series data with temporal dependencies and long-term memory. Two sources of such data are CoinDesk and Yahoo Finance.
- Best input: Historical prices and related variables as input features.
- Best output: Prediction of future prices.
- Models that can be combined: ARIMA, SVR, and Random Forest models.


## Heterogeneous agent model developed by Brock and Hommes, 1998

The first model we use is based on the standard framework for estimating the speculative bubble component in asset prices. This model assumes that asset prices can be decomposed into two components: a fundamental component and a speculative component. The fundamental component is driven by the intrinsic value of the asset, while the speculative component is driven by market sentiment and investors' expectations.

To estimate the fundamental component of Bitcoin prices, we use a range of economic indicators, including the hash rate, transaction volume, and mining difficulty. We also consider the macroeconomic environment, such as inflation rates and interest rates, to account for the broader economic context in which Bitcoin operates.

To estimate the speculative component of Bitcoin prices, we use a variety of technical indicators, including moving averages, relative strength index (RSI), and the stochastic oscillator. We also use sentiment analysis of social media and news articles to gauge market sentiment and investor expectations.

The Heterogeneous agent model developed by Brock and Hommes (1998) assumes that the asset price $P_t$ can be decomposed into a fundamental component $F_t$ and a speculative component $S_t$ as follows:

$$P_t = F_t + S_t$$

The fundamental component of Bitcoin prices can be estimated using the following equation:

$$F_t = \omega_0 + \sum_{j=1}^{N} \omega_j X_{j,t}$$

where $F_t$ is the fundamental component of Bitcoin prices at time $t$, $X_{j,t}$ are the **economic indicators** and **macroeconomic factors** at time $t$, $N$ is the total number of indicators and factors, and $\omega_j$ are the corresponding weights.

$$ F_t = \omega_0 + w_1 \cdot \text{hash rate}_t + w_2 \cdot \text{transaction volume}_t + w_3 \cdot \text{mining difficulty}_t + w_4 \cdot \text{inflation rates}_t + w_5 \cdot \text{interest rates}_t $$

Where:

- $F_t$ is the fundamental component of Bitcoin prices at time $t$
- $\text{hash rate}_t$ is the hash rate of the Bitcoin network at time $t$
- $\text{transaction volume}_t$ is the transaction volume on the Bitcoin network at time $t$
- $\text{mining difficulty}_t$ is the mining difficulty of the Bitcoin network at time $t$
- $\text{inflation rates}_t$ is the inflation rate at time $t$
- $\text{interest rates}_t$ is the interest rate at time $t$
- $w_1, w_2, w_3, w_4,$ and $w_5$ are weights assigned to each of the economic indicators and macroeconomic factors, respectively.

The speculative component of Bitcoin prices can be estimated using the following equation:

$$S_t = \sum_{j=1}^{M} \alpha_j Y_{j,t} + \beta S_{t-1}$$

where $S_t$ is the speculative component of Bitcoin prices at time $t$, $Y_{j,t}$ are the **technical indicators** and **sentiment analysis** at time $t$, $M$ is the total number of technical indicators and sentiment analysis, $\alpha_j$ are the corresponding weights, and $\beta$ is the persistence parameter.

$Y_{j,t}$, which represents the $j$th technical indicator or sentiment analysis at time $t$, can be written as:

$$Y_{j,t} = f_j (P_t, V_t, M_t, N_t, S_t, A_t, E_t)$$

where $P_t$ is the price of Bitcoin at time $t$, $V_t$ is the trading volume of Bitcoin at time $t$, $M_t$ is the mining difficulty of Bitcoin at time $t$, $N_t$ is the number of active Bitcoin nodes at time $t$, $S_t$ is the market sentiment of Bitcoin at time $t$, $A_t$ is the adoption rate of Bitcoin at time $t$, and $E_t$ is the external news and events related to Bitcoin at time $t$. The function $f_j$ represents the specific technical indicator or sentiment analysis being used, and may have different inputs and parameters depending on the indicator.

For example, the formula for the **moving average** indicator ($MA$) with a window size of $k$ can be written as:

$$Y_{MA,t} = \frac{1}{k} \sum_{i=t-k+1}^{t} P_i$$

where $P_i$ is the price of Bitcoin at time $i$.

Similarly, the formula for the **relative strength index** ($RSI$) with a window size of $k$ can be written as:

$$Y_{RSI,t} = 100 - \frac{100}{1 + RS}$$

where $RS$ is the relative strength at time $t$, which is calculated as:

$$RS = \frac{\sum_{i=t-k+1}^{t} Max(P_i - P_{i-1}, 0)}{\sum_{i=t-k+1}^{t} |P_i - P_{i-1}|}$$

The formula for the **stochastic oscillator** ($SO$) with a window size of $k$ can be written as:

$$Y_{SO,t} = \frac{P_t - Min_{k}(P)}{Max_{k}(P) - Min_{k}(P)} \times 100$$

where $Min_{k}(P)$ and $Max_{k}(P)$ are the minimum and maximum prices of Bitcoin over the past $k$ periods, respectively.

The **sentiment analysis** indicator ($SA$) at time $t$ can be written as:

$$Y_{SA,t} = f_{SA}(T_t, A_t, E_t)$$

where $T_t$ is the text data extracted from news articles and social media related to Bitcoin at time $t$, and $f_{SA}$ is a function that processes the text data to generate a sentiment score. The sentiment score may be based on techniques such as keyword analysis, natural language processing, or machine learning.

> **This model can be useful for predicting how different types of market participants, such as traders and investors, may behave and how this may affect Bitcoin prices.**

### Model training data, input data, output data, integrations
- Data: Time series data with behavioral aspects of market participants. Two sources of such data are BitcoinTalk forum and Twitter.
- Best input: Historical prices and sentiment data as input features.
- Best output: Prediction of future prices and identification of market regimes.
- Models that can be combined: VAR, MSGARCH, and LSTM models.

## Shiller's cyclically adjusted price-to-earnings ratio (CAPE) Ratio, 2000

Shiller's CAPE ratio has been applied to the valuation of Bitcoin in the literature. However, it is important to note that the applicability of traditional stock market valuation models, such as the CAPE ratio, to the cryptocurrency market is still a matter of debate and further research is needed to determine their effectiveness in predicting Bitcoin prices.

The cyclically adjusted price-to-earnings (CAPE) ratio, also known as the Shiller PE ratio, is a valuation measure that uses real earnings per share over a 10-year period to smooth out fluctuations in corporate profits that occur over different periods of the business cycle. The formula for calculating the CAPE ratio is as follows:

$$CAPE = \frac{P}{E_{10}}$$

where,

$P$ is the price of the asset
$E_{10}$ is the average of the inflation-adjusted earnings of the asset over the previous 10 years

> **This model can be useful for predicting when the Bitcoin market may be overvalued or undervalued based on historical price data.**

### Model training data, input data, output data, integrations

- Data: Time series data of the cyclically-adjusted price-to-earnings (CAPE) ratio. Two sources of such data are Yale University and FRED.
- Best input: Historical CAPE ratio as input feature.
- Best output: Prediction of future prices.
- Models that can be combined: ARIMA, VAR, and Random Forest models.

## Markov switching GARCH (MSGARCH) models: Developed by Robert Engle and Kevin Sheppard in the 2000s

The MSGARCH model combines the GARCH model with a Markov regime-switching model to account for changes in volatility over time. The MSGARCH model assumes that the variance of the time series is dependent on the state of an unobservable Markov chain. The state of the Markov chain determines the volatility of the time series.

Let $y_t$ be a $K \times 1$ vector of variables at time $t$, and $h_t$ be the conditional variance-covariance matrix of $y_t$ given the information set $F_{t-1}$. The MSGARCH model can be written as:

$$h_t = A_{s_t} + \sum_{i=1}^p B_i y_{t-i} y_{t-i}' B_i' + \sum_{j=1}^q C_j h_{t-j} C_j'$$

where $s_t$ is the state of the Markov chain at time $t$, $A_{s_t}$ is a $K \times K$ matrix that contains the unconditional variance-covariance matrix of $y_t$ in state $s_t$, and $B_i$ and $C_j$ are coefficient matrices.

> **MSGARCH can be useful for predicting changes in market regimes and how they may affect the volatility of Bitcoin prices.**

### Model training data, input data, output data, integrations
- Data: Historical price data of Bitcoin, including volatility measures such as GARCH residuals
- Sources: Cryptocurrency exchanges, financial data providers such as Bloomberg or Yahoo Finance
- Best input: Historical price data of Bitcoin and other relevant cryptocurrencies, as well as relevant macroeconomic indicators that may affect the cryptocurrency market
- Best output: Predictions of future volatility of Bitcoin and other cryptocurrencies
- Combination models: Random Forest, LSTM, SVR

## Random Forest model: Developed by Leo Breiman in 2001

The random forest model is an ensemble learning algorithm used for classification and regression analysis. It works by constructing a multitude of decision trees at training time and outputting the class or mean prediction of the individual trees. The random forest model reduces overfitting by aggregating the results of many decision trees.

The mathematical equation for the random forest model is as follows:

Given a training set ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$, where $x_i$ is the input data and $y_i$ is the corresponding output, the goal is to find a function $f(x)$ that predicts the output y for a given input $x$.

The random forest model constructs $M$ decision trees, each of which is trained on a bootstrapped sample of the training data. At each node of a decision tree, a random subset of the input features is considered for splitting.

The prediction of the random forest model is obtained by averaging the predictions of the individual decision trees:

$$f(x) = \frac{1}{M} \sum_{j=1}^M f_j(x)$$

where $f_j(x)$ is the prediction of the $j-th$ decision tree.

> **Random Forest can be useful for predicting Bitcoin prices based on a large set of input data, where the relationships between the input and output variables may be nonlinear and difficult to model.**

### Model training data, input data, output data, integrations

- Data: Historical price data of Bitcoin, including various technical indicators such as moving averages, relative strength index (RSI), and stochastic oscillator
- Sources: Cryptocurrency exchanges, technical analysis platforms such as TradingView or Coinigy
- Best input: Historical price data of Bitcoin and other relevant cryptocurrencies, as well as relevant macroeconomic indicators that may affect the cryptocurrency market
- Best output: Predictions of future price movements of Bitcoin and other cryptocurrencies
- Combination models: LSTM, SVR, MSGARCH

## ARIMA-NN, a Time series forecasting using a hybrid ARIMA and neural network model by Zhang, G. P., 2003

The hybrid ARIMA and neural network model proposed by Zhang (2003) combines the strengths of the Autoregressive Integrated Moving Average (ARIMA) model and neural network models to improve the accuracy of time series forecasting. The model is based on the following equation:

$$\hat{y}{t+h|t}=y_t + \sum{i=1}^{p}\phi_i(y_{t+1-i}-y_t) + \sum_{j=1}^{q}\theta_j\varepsilon_{t+1-j} + f(y_t, y_{t-1},..., y_{t-p}, \varepsilon_{t-1}, \varepsilon_{t-2}, ..., \varepsilon_{t-q})$$

where $\hat{y}_{t+h|t}$ is the forecast for time $t+h$, given data up to time $t$; $p$ and $q$ are the orders of the autoregressive and moving average parts of the ARIMA model, respectively; $\phi_i$ and $\theta_j$ are the corresponding coefficients; $\varepsilon_t$ is the error term; and $f$ is a neural network model that captures any nonlinear patterns in the data.

The neural network component of the model is trained using a backpropagation algorithm to minimize the mean squared error between the actual and predicted values. The weights of the neural network are updated iteratively during the training process until convergence.

The hybrid ARIMA and neural network model has been shown to outperform traditional time series models in many applications, including forecasting stock prices, exchange rates, and electricity demand.

> **This model can be useful for predicting Bitcoin prices based on a combination of time series analysis and machine learning techniques.**

### Model training data, input data, output data, integrations
- Data: Historical price data of Bitcoin
- Sources: Cryptocurrency exchanges, financial data providers such as Bloomberg or Yahoo Finance
- Best input: Historical price data of Bitcoin and other relevant cryptocurrencies, as well as relevant macroeconomic indicators that may affect the cryptocurrency market
- Best output: Predictions of future price movements of Bitcoin and other cryptocurrencies
- Combination models: Random Forest, LSTM, SVR

## Modified version of the model proposed by Phillips et al., 2011

The second model we use is based on the behavioral finance literature, which suggests that investors' irrational behavior can create speculative bubbles in financial markets.

This model assumes that market sentiment and investor behavior are driven by a range of psychological biases, including herding behavior, overconfidence, and confirmation bias.

The modified Phillips et al. (2011) model for detecting speculative bubbles in financial markets is defined by the following equation:

$$y_t = \beta_0 + \beta_1x_{1,t} + \beta_2x_{2,t} + \beta_3x_{3,t} + \epsilon_t$$

where,

$y_t$ is the log of the price of the asset at time $t$
$x_{1,t}$ is the log of the ratio of the market capitalization of the asset to the market capitalization of all cryptocurrencies
$x_{2,t}$ is the Google Trends search volume index for the term "Bitcoin"
$x_{3,t}$ is the number of Bitcoin-related tweets and Reddit posts
$\beta_0, \beta_1, \beta_2,$ and $\beta_3$ are the parameters of the model
$\epsilon_t$ is the error term.

To estimate the speculative bubble component of Bitcoin prices using this model, we use a range of behavioral finance indicators, including the ratio of Bitcoin to the market capitalization of all cryptocurrencies, the Google Trends search volume index for the term "Bitcoin," and the number of Bitcoin-related tweets and Reddit posts.

> **This modified version of the model may be useful for predicting Bitcoin prices based on a combination of statistical and machine learning techniques.**

### Model training data, input data, output data, integrations
- Data: Historical price data of Bitcoin, as well as various technical indicators such as moving averages, RSI, and MACD
- Sources: Cryptocurrency exchanges, technical analysis platforms such as TradingView or Coinigy
- Best input: Historical price data of Bitcoin and other relevant cryptocurrencies, as well as relevant macroeconomic indicators that may affect the cryptocurrency market
- Best output: Predictions of future price movements of Bitcoin and other cryptocurrencies
- Combination models: LSTM, SVR, MSGARCH

## 1930s-1940s, Elliott Wave Model, Ralph Nelson Elliott

The Elliott Wave Model is a technical analysis tool that attempts to predict financial market trends by identifying recurring patterns in market prices. The model was developed by Ralph Nelson Elliott in the 1930s-1940s and is based on the idea that markets move in waves of varying degrees. These waves are composed of two types of movements: impulse waves and corrective waves. Impulse waves are trends that move in the direction of the overall market trend, while corrective waves move against the trend. The Elliott Wave Model can help predict the bitcoin price by analyzing past price movements and identifying these wave patterns in the current market. Traders can use this information to make decisions on when to buy or sell bitcoin based on the expected direction of the market trend.

## 1936, Liquidity Preference Theory, John Maynard Keynes

The Liquidity Preference Theory was developed by John Maynard Keynes in 1936 and is an economic theory that explains how interest rates are determined in the market. The theory proposes that interest rates are influenced by the supply and demand for money, with the demand for money being determined by the desire to hold liquid assets, such as cash and short-term securities. The Liquidity Preference Theory can help predict the bitcoin price by analyzing changes in interest rates and the demand for money. As demand for money increases, interest rates are likely to rise, which can affect the price of bitcoin and other financial assets.



## 1938, Dividend Discount Model (DDM), John Burr Williams

The Dividend Discount Model (DDM) is a financial model developed by John Burr Williams in 1938 that calculates the intrinsic value of a stock based on its expected future dividend payments. The model assumes that the value of a stock is equal to the present value of its future dividend payments, discounted at a certain rate. The DDM can help predict the bitcoin price by analyzing changes in the expected future dividend payments of bitcoin-related stocks or companies. Traders can use this information to make decisions on whether to invest in these stocks or companies based on their expected future returns.



## 1940s-1950s, Monte Carlo Simulation Models, Stanislaw Ulam, John von Neumann

Monte Carlo Simulation Models are mathematical models that use random sampling techniques to simulate the outcomes of complex systems. The models were developed by Stanislaw Ulam and John von Neumann in the 1940s-1950s and have been widely used in finance to simulate the behavior of financial markets. Monte Carlo Simulation Models can help predict the bitcoin price by simulating the outcomes of different market scenarios and analyzing their potential impact on the price of bitcoin. This can help traders make informed decisions on how to manage their investments and minimize their risk exposure.



## 1952, Markowitz's Portfolio Theory, Harry Markowitz

Markowitz's Portfolio Theory is a financial theory developed by Harry Markowitz in 1952 that proposes that investors can minimize their risk exposure by diversifying their investment portfolios across different asset classes. The theory is based on the idea that combining assets with low correlations can reduce overall portfolio risk. Markowitz's Portfolio Theory can help predict the bitcoin price by analyzing the impact of changes in market conditions on the risk exposure of bitcoin-related investments. Traders can use this information to diversify their portfolios and manage their risk exposure in different market conditions.

## 1953, Random Walk Model, Maurice Kendall

The random walk model is a theory that assumes that stock prices change randomly over time and that future prices cannot be predicted based on past prices. In other words, the model suggests that the price movements of a stock are independent of each other and follow a random path. This model can help predict the Bitcoin price by assuming that the future price of Bitcoin will be the same as the current price, and the direction of future price movements is unpredictable. Therefore, the model suggests that it is not possible to predict the future price of Bitcoin based on past price movements.

## 1956, Adaptive Expectations Hypothesis, Phillip Cagan

The adaptive expectations hypothesis is an economic theory that suggests that people adjust their expectations of future events based on past experiences. In other words, the hypothesis suggests that individuals make predictions about future events based on past trends and events. This model can help predict the Bitcoin price by assuming that the future price of Bitcoin will be influenced by past price movements, and individuals will adjust their expectations of the future price of Bitcoin based on these past movements. Therefore, the model suggests that it is possible to predict the future price of Bitcoin by analyzing past price movements and trends.

## 1956, Gordon Growth Model, Myron Gordon, Eli Shapiro

The Gordon growth model is a stock valuation model that assumes that the price of a stock is determined by its future dividends and the growth rate of those dividends. The model suggests that the value of a stock is equal to its expected future dividends divided by the difference between the required rate of return and the growth rate of those dividends. This model can help predict the Bitcoin price by assuming that the price of Bitcoin is determined by its expected future returns and the growth rate of those returns. Therefore, the model suggests that it is possible to predict the future price of Bitcoin by analyzing the expected future returns of Bitcoin and the growth rate of those returns.



## 1958, Modigliani and Miller's Capital Structure model, Franco Modigliani and Merton Miller

Modigliani and Miller's Capital Structure model is an economic theory that suggests that the value of a firm is not affected by its capital structure. The theory suggests that the value of a firm is determined by its cash flows and the riskiness of those cash flows, not by its capital structure. This model can help predict the Bitcoin price by assuming that the value of Bitcoin is not affected by its capital structure. Therefore, the model suggests that it is possible to predict the future price of Bitcoin by analyzing its cash flows and the riskiness of those cash flows.



## 1958, Modigliani-Miller theorem (MM), Franco Modigliani and Merton Miller

Modigliani-Miller theorem is an economic theory that suggests that the value of a firm is not affected by its financing decisions. The theorem suggests that the value of a firm is determined by its cash flows and the riskiness of those cash flows, not by the way it is financed. This model can help predict the Bitcoin price by assuming that the value of Bitcoin is not affected by its financing decisions. Therefore, the model suggests that it is possible to predict the future price of Bitcoin by analyzing its cash flows and the riskiness of those cash flows.



## 1958, Phillips Curve Model, Alban William Phillips

The Phillips Curve model is an economic theory that suggests that there is a trade-off between inflation and unemployment. The theory suggests that as unemployment falls, inflation rises, and vice versa. This model can help predict the Bitcoin price by assuming that the inflation rate will influence the future price of Bitcoin. Therefore, the model suggests that it is possible to predict the future price of Bitcoin by analyzing the inflation rate.



## 1960, Kalman Filter Models, Rudolf Emil Kalman

The Kalman Filter Model is a mathematical algorithm used to predict future events based on past observations. It is a widely used model in control systems engineering, and it uses a series of measurements to estimate the state of a system. In the context of predicting Bitcoin prices, the Kalman Filter Model could be used to estimate the volatility of Bitcoin prices based on past price movements and market trends. The model could help identify trends in the Bitcoin market and provide investors with insights into where the price may be heading.



## 1960s-1970s, Efficient Market Hypothesis (EMH), Eugene Fama

The Efficient Market Hypothesis (EMH) is a theory that states that financial markets are efficient, and that asset prices always reflect all available information. In other words, it suggests that it is impossible to beat the market by trying to predict stock prices or other asset prices. In the context of predicting Bitcoin prices, the EMH could suggest that it is not possible to predict the future price of Bitcoin based on past prices or market trends. Therefore, the EMH could suggest that investors should focus on the fundamentals of the cryptocurrency and not try to time the market.



## 1960s, Capital Asset Pricing Model (CAPM), William Sharpe, John Lintner, and Jan Mossin

The Capital Asset Pricing Model (CAPM) is a model used to calculate the expected return on an asset based on its risk and the expected return on the market as a whole. The model assumes that investors are rational and that they only invest in assets that have a positive expected return. In the context of predicting Bitcoin prices, the CAPM could be used to estimate the expected return on Bitcoin based on its risk level and the expected return on the overall cryptocurrency market.



## 1961, Miller and Modigliani's Dividend Irrelevance Theory, Merton Miller and Franco Modigliani

Miller and Modigliani's Dividend Irrelevance Theory is a theory that suggests that the dividend policy of a company does not affect the overall value of the company. In other words, investors are indifferent to the payment of dividends, and the value of the company is based on its future earnings potential. In the context of predicting Bitcoin prices, the theory could suggest that the future value of Bitcoin is based on its adoption and acceptance as a means of payment and store of value, and not on its dividend policy.

## 1963, Fractal Analysis, Benoit Mandelbrot

Fractal Analysis is a mathematical technique used to analyze complex patterns and structures in data. It is used to identify patterns that repeat at different scales, and it has been used to analyze financial markets. In the context of predicting Bitcoin prices, Fractal Analysis could be used to identify patterns in the price movements of Bitcoin and to predict future price movements based on these patterns.

## 1970, Akerlof's Market for Lemons model, George Akerlof

Akerlof's Market for Lemons Model is a model used to explain the problems that arise when there is asymmetric information in a market. The model suggests that when buyers cannot distinguish between high-quality and low-quality goods, the market may become dominated by low-quality goods, or "lemons". In the context of predicting Bitcoin prices, the model could suggest that the market for Bitcoin may be dominated by low-quality coins or fraudulent activity, which could affect the overall value of the cryptocurrency.

## 1972, Rational Expectations Hypothesis, Robert Lucas Jr.

The Rational Expectations Hypothesis is a theory that suggests that people make decisions based on all available information and that they have rational expectations about future events. In the context of predicting Bitcoin prices, the theory could suggest that the market price of Bitcoin already reflects all available information and that it is impossible to gain an advantage by trying to predict the future price of the cryptocurrency.



## 1973, Black-Scholes Option Pricing Model, Fischer Black and Myron Scholes

The Black-Scholes model is a mathematical model used to estimate the price of financial instruments such as options. It assumes that the price of the underlying asset follows a geometric Brownian motion with constant drift and volatility. The model uses several inputs, including the current price of the underlying asset, the option's strike price, the time until expiration, and the risk-free interest rate. The model can be used to predict the theoretical value of an option at any point in time.

The Black-Scholes model can be used to predict the price of bitcoin options. Traders and investors can use the model to estimate the fair value of an option and compare it to the market price. If the market price is below the theoretical value, the option is considered undervalued and may represent a buying opportunity. Conversely, if the market price is above the theoretical value, the option is considered overvalued and may represent a selling opportunity.



## 1973, Merton Model, Robert Merton

The Merton model is a mathematical model used to estimate the credit risk of a company or individual. It assumes that the value of a firm's assets follows a stochastic process and that the firm defaults when the value of its assets falls below a certain threshold. The model can be used to estimate the probability of default and the expected loss in the event of default.

The Merton model can be used to predict the credit risk of companies that hold bitcoin on their balance sheets. By estimating the probability of default and the expected loss, investors and creditors can assess the creditworthiness of the company and adjust their investment or lending decisions accordingly.

## 1973, Random Walk Theory, Burton Malkiel

The Random Walk Theory states that stock prices and other financial market prices are inherently unpredictable and follow a random walk. The theory suggests that past price movements cannot be used to predict future price movements and that the best strategy for investors is to buy and hold a diversified portfolio of assets.

The Random Walk Theory can be used to argue that predicting the price of bitcoin is inherently difficult and that investors should focus on building a diversified portfolio of assets rather than trying to time the market.

## 1974, Barro's Ricardian Equivalence Model, Robert Barro

The Ricardian Equivalence Model suggests that changes in government spending have no effect on the economy because individuals anticipate future tax increases to pay for the government spending and adjust their behavior accordingly. The model assumes that individuals are forward-looking and rational and that they save more in anticipation of future tax increases.

The Ricardian Equivalence Model can be used to argue that changes in government policy related to bitcoin, such as increased regulation or taxation, are unlikely to have a significant impact on the bitcoin price because investors will adjust their behavior accordingly.

## 1974, Merton's Structural Credit Risk Model, Robert Merton

The Merton Structural Credit Risk Model is a variation of the Merton model that takes into account the possibility of multiple sources of default risk. The model assumes that a company can default if any one of its underlying assets falls below a certain threshold. The model can be used to estimate the probability of default and the expected loss in the event of default.

The Merton Structural Credit Risk Model can be used to predict the credit risk of companies that hold bitcoin on their balance sheets. By estimating the probability of default and the expected loss, investors and creditors can assess the creditworthiness of the company and adjust their investment or lending decisions accordingly.

## 1976, Arbitrage Pricing Theory (APT), Stephen Ross

APT is a model used in finance to determine the relationship between asset prices and their underlying economic factors. It suggests that the price of an asset should reflect its risk premium, which is influenced by various factors such as interest rates, inflation, and market volatility. APT can help predict the bitcoin price by analyzing the impact of these economic factors on the cryptocurrency market.



## 1976, Jensen and Meckling's Agency Theory, Michael Jensen and William Meckling

Agency theory is a model used to explain the relationship between principals (such as shareholders) and agents (such as managers) in organizations. It suggests that agents may have different objectives than principals and that conflicts may arise between them. This theory can help predict the bitcoin price by analyzing how the interests of different stakeholders in the cryptocurrency market may affect its price.



## 1976, Jump-diffusion models, Robert Merton

Jump-diffusion models are used to describe the behavior of asset prices that have both continuous and sudden jumps. It can help predict the bitcoin price by analyzing the impact of unexpected events or news on the cryptocurrency market.



## 1977, Real options theory, Stewart Myers

Real options theory is a model used to evaluate investment opportunities by analyzing the value of flexibility and the ability to make decisions in uncertain environments. It can help predict the bitcoin price by analyzing the potential impact of various events or decisions on the cryptocurrency market.



## 1977, Vasicek Model, Oldrich Vasicek

The Vasicek model is a type of stochastic process used to model interest rate movements over time. It can help predict the bitcoin price by analyzing the impact of changes in interest rates on the cryptocurrency market.



## 1979, Binomial Option Pricing Model, Cox-Ross-Rubinstein

The binomial option pricing model is a mathematical method used to calculate the value of options based on the probability of different future outcomes. It can help predict the bitcoin price by analyzing the impact of various potential future scenarios on the cryptocurrency market.



## 1980, Grossman and Stiglitz's Information Asymmetry model, Sanford Grossman and Joseph Stiglitz

The Information Asymmetry model is a theory that explains how differences in access to information between market participants can lead to inefficiencies in markets. It can help predict the bitcoin price by analyzing the impact of information asymmetry on the cryptocurrency market.



## 1980, Hansen and Sargent's Uncertainty Models, Lars Hansen and Thomas Sargent

Uncertainty models are used to model decision-making under conditions of uncertainty. It can help predict the bitcoin price by analyzing how market participants are likely to react to uncertain events in the cryptocurrency market.


## 1981, International capital asset pricing model (ICAPM), René Stulz

The ICAPM is a model used to determine the relationship between asset prices and their underlying economic factors in a global context. It can help predict the bitcoin price by analyzing the impact of global economic factors on the cryptocurrency market.

## 1982, Real Business Cycle Theory, Finn Kydland and Edward Prescott

Real Business Cycle Theory is a macroeconomic theory that explains how fluctuations in productivity and other economic factors can lead to business cycles. It can help predict the bitcoin price by analyzing the impact of macroeconomic factors on the cryptocurrency market.

## 1983, Diamond and Dybvig's Bank Run Model, Douglas Diamond and Philip Dybvig

The Bank Run Model is a model used to explain the occurrence of bank runs and the impact of deposit insurance on the stability of the banking system. It can help predict the bitcoin price by analyzing the potential for runs on exchanges or wallets that hold large amounts of bitcoin.



## 1985, Cox-Ingersoll-Ross (CIR) model, John Cox, Jonathan Ingersoll, and Stephen Ross

The Cox-Ingersoll-Ross (CIR) model is a stochastic differential equation used to model interest rates. The model assumes that the short-term interest rate follows a mean-reverting process with constant volatility. It can be used to forecast the volatility of interest rates and help predict changes in the interest rates which can have an impact on the bitcoin price.



## 1985, New Keynesian Economics, N. Gregory Mankiw

New Keynesian Economics is a macroeconomic model that incorporates rational expectations and price stickiness. The model assumes that markets are not always efficient and that government intervention may be necessary to stabilize the economy. This model can help predict the impact of changes in government policies on the overall economy and how it can affect the demand for bitcoin.



## 1986, Ho-Lee Model, Thomas Ho, Sang Bin Lee

The Ho-Lee Model is a stochastic interest rate model used to forecast interest rate movements. The model assumes that interest rates are driven by a combination of market factors and other economic variables. This model can be used to forecast changes in the interest rates which can have an impact on the bitcoin price.



## 1989, Wavelet Analysis, Stephane Mallat

Wavelet Analysis is a mathematical technique used to analyze and predict time series data. The technique decomposes a time series into different frequency components, which allows for the identification of patterns and trends that are not easily visible in the raw data. This model can be used to analyze the historical price movements of bitcoin and identify patterns that may indicate future price movements.



## 1990s-2000s, Network Analysis Models, Mark Newman, Albert-László Barabási, Duncan J. Watts

Network Analysis Models are used to model complex systems using graph theory. These models are used to identify the relationships between different components of the system and how they interact with each other. In the context of bitcoin, these models can be used to analyze the network of bitcoin transactions and identify patterns that may indicate future price movements.



## 1992, Fama-French Three-Factor Model, Eugene Fama, Kenneth French

The Fama-French Three-Factor Model is an asset pricing model that takes into account three factors: market risk, size, and value. The model is used to explain the returns on a portfolio of stocks or other assets. This model can be used to analyze the bitcoin market and identify the factors that drive its returns.



## 1992, Minsky Model of Financial Instability, Hyman Minsky

The Minsky Model of Financial Instability is a macroeconomic model that explains the dynamics of financial crises. The model assumes that periods of stability in the financial markets lead to excessive risk-taking, which eventually leads to a financial crisis. This model can be used to predict the likelihood of a financial crisis in the bitcoin market.



## 1993, Heston Model, Steven Heston

The Heston Model is a stochastic volatility model used to forecast asset prices. The model assumes that the volatility of an asset is itself a stochastic process that is driven by a combination of market factors and other economic variables. This model can be used to forecast changes in the volatility of the bitcoin price.



## 1993, Metcalfe's Law Model, Robert Metcalfe

Metcalfe's Law Model is a network effect model that states that the value of a network is proportional to the square of the number of its users. In the context of bitcoin, this model can be used to predict the price of bitcoin based on the number of active bitcoin users.



## 1993, Stochastic volatility models, Steven Heston

Stochastic volatility models are used to forecast the volatility of financial assets. These models assume that the volatility of an asset is itself a stochastic process that is driven by a combination of market factors and other economic variables. This model can be used to forecast changes in the volatility of the bitcoin price.



## 1993, Taylor's Monetary Policy Rules, John Taylor

This model provides a framework for policymakers to make decisions regarding monetary policy, including setting interest rates, based on a set of rules rather than discretion. The model considers factors such as inflation, output, and the equilibrium interest rate. It can help predict the bitcoin price by providing insight into how changes in monetary policy might impact the broader economy and financial markets, including bitcoin.

## 1994, Long-term capital management (LTCM) model, Robert Merton and Myron Scholes

This model was developed by a hedge fund that used a quantitative approach to investing. The model is based on the premise that markets are efficient and that there is a mathematical relationship between different asset classes. It can help predict the bitcoin price by providing insights into how different asset classes, such as stocks, bonds, and currencies, might impact the price of bitcoin.

## 1996, Log-Periodic Power Law (LPPL) Model, Didier Sornette

This model is based on the idea that financial markets experience a predictable pattern of oscillations before experiencing a crash or bubble. It can help predict the bitcoin price by identifying patterns in the price of bitcoin that may be indicative of an impending crash or bubble.



## 1997, Robert Shiller, Daniel Kahneman, Amos Tversky, Richard Thaler, Behavioral Finance Models

These models consider how human psychology and behavior can impact financial markets. They are based on the idea that individuals may not always make rational decisions when it comes to investing and that emotions and biases can play a role in decision-making. They can help predict the bitcoin price by providing insights into how investor sentiment and market psychology might impact the price of bitcoin.

## 1999, Asset pricing models with habit formation, John Campbell and John Cochrane

This model considers how individuals form habits when it comes to investing and how these habits can impact asset prices. It can help predict the bitcoin price by providing insights into how changes in investor habits might impact the price of bitcoin.

## 1999, Bernanke and Gertler's Financial Accelerator Model, Ben Bernanke and Mark Gertler

This model considers how changes in the availability of credit can impact the broader economy and financial markets. It can help predict the bitcoin price by providing insights into how changes in credit availability might impact investor sentiment and the price of bitcoin.

## 1999, Informational overshooting, booms, and crashes, Zeira

This model is based on the idea that markets can experience periods of excessive optimism or pessimism that are not supported by fundamental factors. It can help predict the bitcoin price by providing insights into how market sentiment and speculation might impact the price of bitcoin.

## 2000, Johansen-Ledoit-Sornette (JLS) Model, Peter Johansen, Olivier Ledoit, Didier Sornette

This model is based on the idea that financial bubbles can be identified by looking at the price and volume of trading in a market. It can help predict the bitcoin price by providing insights into how patterns in trading volume and price might be indicative of a bubble or impending crash.

## 2000, Predicting financial crashes using discrete scale invariance, Johansen and Sornette

This model is based on the idea that financial crashes can be predicted by looking at the fractal patterns in market data. It can help predict the bitcoin price by providing insights into how patterns in market data might be indicative of an impending crash.

## 2000, The confidence index model, Shiller

This model considers how changes in investor confidence can impact asset prices. It can help predict the bitcoin price by providing insights into how changes in investor confidence might impact the price of bitcoin.

## 2001, Dynamic Conditional Correlation (DCC) Model, Robert Engle, Kevin Sheppard

This model considers how the correlation between different asset classes can change over time. It can help predict the bitcoin price by providing insights into how changes in the correlation between bitcoin and other asset classes might impact the price of bitcoin.

## 2001, Shocks, crashes and bubbles in financial markets, Johansen and Sornette

This model proposes that financial markets are prone to shocks, crashes, and bubbles due to herding behavior of investors and positive feedback loops. It suggests that the occurrence of such events can be predicted by analyzing the behavior of asset prices over time and identifying patterns of positive feedback loops. In the context of Bitcoin, this model can help predict the occurrence of price bubbles and subsequent crashes by analyzing the historical behavior of Bitcoin prices and identifying patterns of herding behavior among investors.



## 2002, The bubble of the millennium: a diachronic analysis of the S&P 500 index, Johansen and Sornette

This model analyzes the behavior of the S&P 500 index to identify patterns of bubble formation and subsequent crashes. It proposes that bubbles are formed due to the herding behavior of investors and can be predicted by analyzing the behavior of asset prices over time. In the context of Bitcoin, this model can help predict the formation of bubbles and subsequent crashes by analyzing the historical behavior of Bitcoin prices and identifying patterns of herding behavior among investors.



## 2003, The volatility feedback model, Shiller

This model proposes that asset prices are influenced by changes in volatility and suggests that market participants use past volatility to forecast future volatility. In the context of Bitcoin, this model can help predict changes in Bitcoin prices by analyzing the historical volatility of Bitcoin prices and identifying patterns of feedback loops between price and volatility.



## 2004, Fearless versus fearful speculative financial bubbles, Andersen and Sornette

This model suggests that bubbles can be classified as either "fearless" or "fearful" based on the behavior of investors. "Fearless" bubbles are characterized by investors who are optimistic about future returns and take on high levels of risk, while "fearful" bubbles are characterized by investors who are pessimistic about future returns and are cautious in their investment decisions. In the context of Bitcoin, this model can help predict the behavior of investors during bubble formation and subsequent crashes, by identifying patterns of risk-taking behavior among investors.



## 2004, Stock market crashes are outliers, Johansen and Sornette

This model suggests that stock market crashes are not random events but are instead the result of a systemic instability in the market. It proposes that such events can be predicted by analyzing the behavior of asset prices over time and identifying patterns of positive feedback loops. In the context of Bitcoin, this model can help predict the occurrence of market crashes by analyzing the historical behavior of Bitcoin prices and identifying patterns of positive feedback loops.



## 2005, Finite-Time Singularity (FTS) Model, Domenico Delli Gatti

This model proposes that financial markets are characterized by "singularities" or moments of extreme volatility, which are caused by the interaction of market participants. It suggests that such events can be predicted by analyzing the behavior of asset prices over time and identifying patterns of extreme volatility. In the context of Bitcoin, this model can help predict the occurrence of extreme price movements by analyzing the historical behavior of Bitcoin prices and identifying patterns of extreme volatility.



## 2008, Network Models of Financial Contagion, Diego Garlaschelli

This model proposes that financial contagion occurs when a shock in one market spreads to other markets through interdependencies between market participants. It suggests that such events can be predicted by analyzing the network of relationships between market participants and identifying patterns of contagion. In the context of Bitcoin, this model can help predict the occurrence of financial contagion by analyzing the network of relationships between Bitcoin market participants and identifying patterns of contagion.



## 2013, Testing for financial crashes using the log-periodic power law model, Bree and Joseph

The log-periodic power law model (LPPL) was proposed in 1997 by Didier Sornette and his colleagues. It is a mathematical model that seeks to explain the behavior of financial markets before a crash. The model postulates that financial crashes are preceded by a bubble-like behavior that can be characterized by a specific mathematical pattern. Specifically, the LPPL model describes how prices deviate from their fundamental value as a function of time. The model predicts that prices will continue to increase at an accelerating rate until they reach a critical point, after which they will start to decline rapidly. The LPPL model can be used to predict when a financial market is likely to crash, and this information can be used to inform investment decisions in the Bitcoin market.



## 2014, LPPL model with mean-reverting residuals, Lin et al.

The LPPL model with mean-reverting residuals was proposed by Lin et al. in 2014. This model extends the original LPPL model by incorporating mean-reverting processes into the residual term. The idea behind this model is that financial markets exhibit both long-term trends and short-term fluctuations that are caused by random noise. The mean-reverting residuals in the LPPL model help to account for these fluctuations, which can be used to make more accurate predictions about the future behavior of the Bitcoin market.



## 2017, The narrative economics model, Shiller

The narrative economics model was proposed by Robert Shiller in 2017. This model suggests that economic trends are driven by the stories that people tell about the economy. These stories can have a powerful effect on people's beliefs and behavior, which in turn can influence the performance of financial markets. The narrative economics model can be used to analyze the impact of news events and media coverage on the Bitcoin market, and to make predictions about future trends based on changes in public sentiment.



## 2019, Stock-to-flow Model, PlanB

The stock-to-flow model was proposed by PlanB in 2019. This model suggests that the price of Bitcoin is largely determined by its scarcity. The model is based on the idea that the value of Bitcoin is proportional to the ratio of its total supply to its annual production rate. In other words, the scarcer Bitcoin becomes, the more valuable it will be. The stock-to-flow model can be used to predict the future price of Bitcoin based on its current stock-to-flow ratio, and to identify periods when Bitcoin is undervalued or overvalued.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

- [Amin Boulouma (@aminblm)](https://github.com/aminblm)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

