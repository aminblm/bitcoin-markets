# Quantifying the Speculative Bubble Component in Bitcoin Markets: A Comparative Analysis of Statistical Models and Empirical Applications

Boulouma A., 2023


## Problem Statement

The problem is to develop a predictive model for the bitcoin market prices that can accurately forecast the price at time $t+1$ based on the price at time $t$.

## Research Question

What is the best predictive model to use for forecasting bitcoin market prices, and what is the predictive power of each model?

## Table of contents


- [Quantifying the Speculative Bubble Component in Bitcoin Markets: A Comparative Analysis of Statistical Models and Empirical Applications](#quantifying-the-speculative-bubble-component-in-bitcoin-markets-a-comparative-analysis-of-statistical-models-and-empirical-applications)
  - [Problem Statement](#problem-statement)
  - [Research Question](#research-question)
  - [Table of contents](#table-of-contents)
  - [Beauty Contest model of Keynes, 1936](#beauty-contest-model-of-keynes-1936)
  - [Artificial Neural Networks (ANN): The concept of artificial neural networks has its roots in the work of Warren McCulloch and Walter Pitts in the 1940s. The development of modern neural networks, including backpropagation, can be attributed to multiple researchers including Paul Werbos in 1974, David Rumelhart and James McClelland in 1986, and Geoffrey Hinton in the 2000s](#artificial-neural-networks-ann-the-concept-of-artificial-neural-networks-has-its-roots-in-the-work-of-warren-mcculloch-and-walter-pitts-in-the-1940s-the-development-of-modern-neural-networks-including-backpropagation-can-be-attributed-to-multiple-researchers-including-paul-werbos-in-1974-david-rumelhart-and-james-mcclelland-in-1986-and-geoffrey-hinton-in-the-2000s)
  - [Bayesian regression model: Bayesian regression has been developed by multiple researchers including Harold Jeffreys in the 1940s and Bruno de Finetti in the 1950s](#bayesian-regression-model-bayesian-regression-has-been-developed-by-multiple-researchers-including-harold-jeffreys-in-the-1940s-and-bruno-de-finetti-in-the-1950s)
  - [Vector autoregression (VAR) model: Introduced by Clive Granger in the 1960s](#vector-autoregression-var-model-introduced-by-clive-granger-in-the-1960s)
  - [ARIMA model: Autoregressive Integrated Moving Average (ARIMA) models: Developed by George Box and Gwilym Jenkins in the 1970s](#arima-model-autoregressive-integrated-moving-average-arima-models-developed-by-george-box-and-gwilym-jenkins-in-the-1970s)
  - [GARCH model: Developed by Robert Engle in the 1980s](#garch-model-developed-by-robert-engle-in-the-1980s)
  - [Markov regime-switching model: Developed by Andrew Harvey in the 1980s](#markov-regime-switching-model-developed-by-andrew-harvey-in-the-1980s)
  - [Support vector regression (SVR) model: Developed by Vladimir Vapnik and Alexey Chervonenkis in the 1990s](#support-vector-regression-svr-model-developed-by-vladimir-vapnik-and-alexey-chervonenkis-in-the-1990s)
  - [The Log-Periodic Power Law (LPPL) by Didier Sornette, 1990](#the-log-periodic-power-law-lppl-by-didier-sornette-1990)
  - [Multivariate GARCH (MGARCH) models: Developed by Robert Engle in the 1990s](#multivariate-garch-mgarch-models-developed-by-robert-engle-in-the-1990s)
  - [Bayesian Neural Networks (BNN): Bayesian neural networks have been developed by multiple researchers including David MacKay in the 1990s and Radford Neal in 1995](#bayesian-neural-networks-bnn-bayesian-neural-networks-have-been-developed-by-multiple-researchers-including-david-mackay-in-the-1990s-and-radford-neal-in-1995)
  - [Long Short-Term Memory (LSTM) models: Developed by Sepp Hochreiter and Jürgen Schmidhuber in 1997](#long-short-term-memory-lstm-models-developed-by-sepp-hochreiter-and-jürgen-schmidhuber-in-1997)
  - [Heterogeneous agent model developed by Brock and Hommes, 1998](#heterogeneous-agent-model-developed-by-brock-and-hommes-1998)
  - [Shiller's CAPE Ratio, 2000](#shillers-cape-ratio-2000)
  - [Markov switching GARCH (MSGARCH) models: Developed by Robert Engle and Kevin Sheppard in the 2000s](#markov-switching-garch-msgarch-models-developed-by-robert-engle-and-kevin-sheppard-in-the-2000s)
  - [Random Forest model: Developed by Leo Breiman in 2001](#random-forest-model-developed-by-leo-breiman-in-2001)
  - [Modified version of the model proposed by Phillips et al., 2011](#modified-version-of-the-model-proposed-by-phillips-et-al-2011)

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


## Artificial Neural Networks (ANN): The concept of artificial neural networks has its roots in the work of Warren McCulloch and Walter Pitts in the 1940s. The development of modern neural networks, including backpropagation, can be attributed to multiple researchers including Paul Werbos in 1974, David Rumelhart and James McClelland in 1986, and Geoffrey Hinton in the 2000s

Artificial neural networks are a type of machine learning algorithm that are modeled after the structure of the human brain. An ANN consists of layers of interconnected nodes, with each node performing a simple mathematical operation. The equations for a feedforward ANN with one hidden layer are:

$$z_j = \sum_{i=1}^n w_{ij} x_i + b_j$$

$$h_j = \sigma(z_j)$$

$$y_k = \sum_{j=1}^m v_{jk} h_j + c_k$$

where $x_i$ is the input to the network, $w_{ij}$ is the weight between the $i$-th input and $j$-th hidden layer node, $b_j$ is the bias of the $j$-th hidden layer node, $z_j$ is the weighted sum of the inputs to the $j$-th hidden layer node, $\sigma(\cdot)$ is the activation function (e.g. sigmoid, ReLU), $h_j$ is the output of the $j$-th hidden layer node, $v_{jk}$ is the weight between the $j$-th hidden layer node and the $k$-th output node, $c_k$ is the bias of the $k$-th output node, and $y_k$ is the output of the network.

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

## Vector autoregression (VAR) model: Introduced by Clive Granger in the 1960s

The VAR model is a time series model that captures the dynamics of multiple time series variables simultaneously. The VAR model can be written as:

$$y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + \epsilon_t$$

where $y_t$ is a $K \times 1$ vector of variables

## ARIMA model: Autoregressive Integrated Moving Average (ARIMA) models: Developed by George Box and Gwilym Jenkins in the 1970s

The ARIMA model is used for time series analysis and forecasting. It is represented as ARIMA(p,d,q), where p is the order of the autoregressive (AR) part, d is the order of differencing, and q is the order of the moving average (MA) part. The equation for an ARIMA model is:

$$\phi_p(B)(1-B)^d X_t = \theta_q(B) \epsilon_t$$

where $X_t$ is the time series, $\epsilon_t$ is the error term, $\phi_p$ and $\theta_q$ are the AR and MA polynomials of order p and q, respectively, $B$ is the backshift operator, and $(1-B)^d$ represents the differencing operation.

## GARCH model: Developed by Robert Engle in the 1980s

The GARCH model is a time series model that captures the dynamics of conditional volatility, where the variance of the error term at time $t$ is a function of the errors at previous times. The GARCH model is specified as follows:

$$y_t = \mu_t + \epsilon_t$$

$$\epsilon_t = \sigma_t z_t$$

$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2$$

where $y_t$ is the time series at time $t$, $\mu_t$ is the conditional mean of the time series, $\epsilon_t$ is the error term at time $t$, $z_t$ is a random variable with a standard normal distribution, $\sigma_t^2$ is the conditional variance of the error term at time $t$, $\omega$ is a constant, $\alpha_i$ and $\beta_j$ are coefficients that determine the weight of the squared error terms and the past variances in the current variance, and $p$ and $q$ are the orders of the autoregressive and moving average terms, respectively.

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

## Multivariate GARCH (MGARCH) models: Developed by Robert Engle in the 1990s

Developed by Robert Engle in the 1990s, the MGARCH model is an extension of the univariate GARCH model to multivariate time series. It models the conditional variance of each series as a function of its own lagged values and the lagged values of the other series in the system. The mathematical equation for a MGARCH(p, q) model is as follows:

$$\boldsymbol{\Sigma}{t} = \boldsymbol{A}{0} + \sum_{i=1}^{p} \boldsymbol{A}{i} \boldsymbol{\epsilon}{t-i} \boldsymbol{\epsilon}{t-i}^{\prime} \boldsymbol{A}{i}^{\prime} + \sum_{j=1}^{q} \boldsymbol{B}{j} \boldsymbol{\Sigma}{t-j} \boldsymbol{B}_{j}^{\prime}$$

where,

$\boldsymbol{\Sigma}_{t}$ is the $k \times k$ covariance matrix of the error terms at time $t$
$\boldsymbol{A}_{0}$ is the $k \times k$ intercept matrix
$\boldsymbol{A}{i}$ and $\boldsymbol{B}{j}$ are $k \times k$ coefficient matrices for the error term and covariance matrix, respectively.
$p$ and $q$ are the order of autoregression and moving average terms for the error term and covariance matrix, respectively.

## Bayesian Neural Networks (BNN): Bayesian neural networks have been developed by multiple researchers including David MacKay in the 1990s and Radford Neal in 1995

The Bayesian neural network model is a type of neural network that incorporates Bayesian inference in the training process. It works by placing a prior distribution over the model parameters and using Bayes' rule to update the posterior distribution of the parameters given the training data. The model uses Monte Carlo methods, such as Markov chain Monte Carlo (MCMC), to approximate the posterior distribution.

The mathematical equation for the Bayesian neural network model is as follows:

Given a training set ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$, where $x_i$ is the input data and $y_i$ is the corresponding output, the goal is to find a function $f(x)$ that predicts the output $y$ for a given input $x$.

The Bayesian neural network model places a prior distribution $p(w)$ over the model parameters $w$ and uses Bayes' rule to update the posterior distribution $p(w|D)$ of the parameters given the training data $D$:

$$p(w|D) = \frac{p(D|w) p(w)}{p(D)}$$

where $p(D|w)$ is the likelihood of the data given the parameters, $p(w)$ is the prior distribution, and $p(D)$ is the marginal likelihood of the data.


## Long Short-Term Memory (LSTM) models: Developed by Sepp Hochreiter and Jürgen Schmidhuber in 1997

LSTM models are a type of recurrent neural network that are designed to capture long-term dependencies in time series data. The equations for an LSTM cell are:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

$$\tilde{C}t = \tanh(W_C x_t + U_C h{t-1} + b_C)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

where $x_t$ is the input at time $t$, $h_{t-1}$ is the previous hidden state, $W_f, U_f, b_f$ are the weights and bias for the forget gate, $W_i, U_i, b_i$ are the weights and bias for the input gate, $W_C, U_C, b_C$ are the weights and bias for the cell state, $W_o, U_o, b_o$ are the weights and bias for the output gate, $\sigma(\cdot)$ is the sigmoid activation function.

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

## Shiller's CAPE Ratio, 2000

Shiller's CAPE ratio has been applied to the valuation of Bitcoin in the literature. However, it is important to note that the applicability of traditional stock market valuation models, such as the CAPE ratio, to the cryptocurrency market is still a matter of debate and further research is needed to determine their effectiveness in predicting Bitcoin prices.

The cyclically adjusted price-to-earnings (CAPE) ratio, also known as the Shiller PE ratio, is a valuation measure that uses real earnings per share over a 10-year period to smooth out fluctuations in corporate profits that occur over different periods of the business cycle. The formula for calculating the CAPE ratio is as follows:

$$CAPE = \frac{P}{E_{10}}$$

where,

$P$ is the price of the asset
$E_{10}$ is the average of the inflation-adjusted earnings of the asset over the previous 10 years

## Markov switching GARCH (MSGARCH) models: Developed by Robert Engle and Kevin Sheppard in the 2000s

The MSGARCH model combines the GARCH model with a Markov regime-switching model to account for changes in volatility over time. The MSGARCH model assumes that the variance of the time series is dependent on the state of an unobservable Markov chain. The state of the Markov chain determines the volatility of the time series.

Let $y_t$ be a $K \times 1$ vector of variables at time $t$, and $h_t$ be the conditional variance-covariance matrix of $y_t$ given the information set $F_{t-1}$. The MSGARCH model can be written as:

$$h_t = A_{s_t} + \sum_{i=1}^p B_i y_{t-i} y_{t-i}' B_i' + \sum_{j=1}^q C_j h_{t-j} C_j'$$

where $s_t$ is the state of the Markov chain at time $t$, $A_{s_t}$ is a $K \times K$ matrix that contains the unconditional variance-covariance matrix of $y_t$ in state $s_t$, and $B_i$ and $C_j$ are coefficient matrices.

## Random Forest model: Developed by Leo Breiman in 2001

The random forest model is an ensemble learning algorithm used for classification and regression analysis. It works by constructing a multitude of decision trees at training time and outputting the class or mean prediction of the individual trees. The random forest model reduces overfitting by aggregating the results of many decision trees.

The mathematical equation for the random forest model is as follows:

Given a training set ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$, where $x_i$ is the input data and $y_i$ is the corresponding output, the goal is to find a function $f(x)$ that predicts the output y for a given input $x$.

The random forest model constructs $M$ decision trees, each of which is trained on a bootstrapped sample of the training data. At each node of a decision tree, a random subset of the input features is considered for splitting.

The prediction of the random forest model is obtained by averaging the predictions of the individual decision trees:

$$f(x) = \frac{1}{M} \sum_{j=1}^M f_j(x)$$

where $f_j(x)$ is the prediction of the $j-th$ decision tree.

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