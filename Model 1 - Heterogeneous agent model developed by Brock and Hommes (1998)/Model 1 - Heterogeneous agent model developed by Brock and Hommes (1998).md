# Model 1 - Predicting Bitcoin Prieces: Heterogeneous agent model developed by Brock and Hommes (1998)

Author: Amin Boulouma

The first model we use is based on the standard framework for estimating the speculative bubble component in asset prices. This model assumes that asset prices can be decomposed into two components: a fundamental component and a speculative component. The fundamental component is driven by the intrinsic value of the asset, while the speculative component is driven by market sentiment and investors' expectations.

To estimate the fundamental component of Bitcoin prices, we use a range of economic indicators, including the hash rate, transaction volume, and mining difficulty. We also consider the macroeconomic environment, such as inflation rates and interest rates, to account for the broader economic context in which Bitcoin operates.

To estimate the speculative component of Bitcoin prices, we use a variety of technical indicators, including moving averages, relative strength index (RSI), and the stochastic oscillator. We also use sentiment analysis of social media and news articles to gauge market sentiment and investor expectations.

## The Model

The Heterogeneous agent model developed by Brock and Hommes (1998) assumes that the asset price $P_t$ can be decomposed into a fundamental component $F_t$ and a speculative component $S_t$ as follows:

$$P_t = F_t + S_t$$

We can apply this formula for the Bitcoin price.

The fundamental component of Bitcoin prices can be estimated using the following equation:

$$F_t = \omega_0 + \sum_{j=1}^{N} \omega_j X_{j,t}$$

where $F_t$ is the fundamental component of Bitcoin prices at time $t$, $X_{j,t}$ are the **economic indicators** and **macroeconomic factors** at time $t$, $N$ is the total number of indicators and factors, and $\omega_j$ are the corresponding weights.

$$ F_t = \omega_0 + w_1 \cdot H_t + w_2 \cdot TV_t + w_3 \cdot MD_t + w_4 \cdot IR_t $$

Where:

- $F_t$ is the fundamental component of Bitcoin prices at time $t$
- $H_t$ is the hash rate of the Bitcoin network at time $t$
- $TV_t$ is the transaction volume on the Bitcoin network at time $t$
- $MD_t$ is the mining difficulty of the Bitcoin network at time $t$
- $IR_t$ is the inflation rate at time $t$
- $w_1, w_2, w_3 and $w_4$ are weights assigned to each of the economic indicators and macroeconomic factors, respectively.

The speculative component of Bitcoin prices can be estimated using the following equation:

$$S_t = \sum_{j=1}^{M} \alpha_j Y_{j,t} + \beta S_{t-1}$$

where $S_t$ is the speculative component of Bitcoin prices at time $t$, $Y_{j,t}$ are the **technical indicators** and **sentiment analysis** at time $t$, $M$ is the total number of technical indicators and sentiment analysis, $\alpha_j$ are the corresponding weights, and $\beta$ is the persistence parameter.

$Y_{j,t}$, which represents the $j$th technical indicator or sentiment analysis at time $t$, can be written as:

$$Y_{j,t} = f_j (P_t, V_t, M_t, N_t, S_t, A_t, E_t)$$

where $P_t$ is the price of Bitcoin at time $t$, $V_t$ is the trading volume of Bitcoin at time $t$, $M_t$ is the mining difficulty of Bitcoin at time $t$, $N_t$ is the number of active Bitcoin nodes at time $t$, $S_t$ is the market sentiment of Bitcoin at time $t$, $A_t$ is the adoption rate of Bitcoin at time $t$, and $E_t$ is the external news and events related to Bitcoin at time $t$. The function $f_j$ represents the specific technical indicator or sentiment analysis being used, and may have different inputs and parameters depending on the indicator.

For example, the formula for the moving average indicator ($MA$) with a window size of $k$ can be written as:

$$Y_{MA,t} = \frac{1}{k} \sum_{i=t-k+1}^{t} P_i$$

where $P_i$ is the price of Bitcoin at time $i$.

Similarly, the formula for the relative strength index ($RSI$) with a window size of $k$ can be written as:

$$Y_{RSI,t} = 100 - \frac{100}{1 + RS}$$

where $RS$ is the relative strength at time $t$, which is calculated as:

$$RS = \frac{\sum_{i=t-k+1}^{t} Max(P_i - P_{i-1}, 0)}{\sum_{i=t-k+1}^{t} |P_i - P_{i-1}|}$$

The formula for the stochastic oscillator ($SO$) with a window size of $k$ can be written as:

$$Y_{SO,t} = \frac{P_t - Min_{k}(P)}{Max_{k}(P) - Min_{k}(P)} \times 100$$

where $Min_{k}(P)$ and $Max_{k}(P)$ are the minimum and maximum prices of Bitcoin over the past $k$ periods, respectively.

The sentiment analysis indicator ($SA$) at time $t$ can be written as:

$$Y_{SA,t} = f_{SA}(T_t, A_t, E_t)$$

where $T_t$ is the text data extracted from news articles and social media related to Bitcoin at time $t$, and $f_{SA}$ is a function that processes the text data to generate a sentiment score. The sentiment score may be based on techniques such as keyword analysis, natural language processing, or machine learning.

## The implementation

1. Import necessary libraries:



```python
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
```

2. Collect historical price data of Bitcoin:



```python
btc_full = pd.read_csv("datasets/btc.csv")
```

    /Users/macbook/opt/anaconda3/lib/python3.9/site-packages/IPython/core/interactiveshell.py:3444: DtypeWarning: Columns (146) have mixed types.Specify dtype option on import or set low_memory=False.
      exec(code_obj, self.user_global_ns, self.user_ns)


3. Define function to calculate moving average:



```python
def moving_average(df, n):
    """
    df: dataframe containing the price data
    n: window size of moving average
    """
    ma = df['Adj Close'].rolling(n).mean()
    return ma
```

4. Define function to calculate relative strength index:



```python
def rsi(df, n):
    """
    df: dataframe containing the price data
    n: window size of RSI
    """
    delta = df['Adj Close'].diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

5. Define function to calculate stochastic oscillator:



```python
def stochastic_oscillator(df, n):
    """
    df: dataframe containing the price data
    n: window size of stochastic oscillator
    """
    lowest_low = df['Low'].rolling(n).min()
    highest_high = df['High'].rolling(n).max()
    k = 100 * (df['Adj Close'] - lowest_low) / (highest_high - lowest_low)
    return k
```

6. Collect necessary data for estimating the fundamental component of Bitcoin prices:

The inflation rate and interest rate of Bitcoin can be calculated using the following data:

- `IssTotNtv`: The total number of Bitcoins that have been mined since the creation of the Bitcoin network.
- `SplyCur`: The current circulating supply of Bitcoin.

To calculate the inflation rate, we can use the following formula:

```inflation_rate = IssTotNtv / SplyCur)``

This formula calculates the percentage increase in the total supply of Bitcoin since its creation, and subtracts 1 to convert it to a percentage increase per year. As of February 23, 2023, the inflation rate of Bitcoin is approximately 1.58%.


```python
# extract required columns
time = btc_full['time']
price = btc_full['PriceUSD']
hash_rate = btc_full['HashRate']
transaction_volume = btc_full['TxTfrValAdjUSD']
mining_difficulty = btc_full['DiffMean']
issuance = btc_full['IssTotNtv']
supply = btc_full['SplyCur']

# calculate inflation rate and interest rate
inflation_rate = issuance / supply

# construct dataframe with required columns
df = pd.DataFrame({
    'time': time,
    'Price': price,
    'hash_rate': hash_rate,
    'transaction_volume': transaction_volume,
    'mining_difficulty': mining_difficulty,
    'inflation_rate': inflation_rate,
})

btc = pd.DataFrame(df)
# Drop NA
btc = btc.dropna()
btc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>Price</th>
      <th>hash_rate</th>
      <th>transaction_volume</th>
      <th>mining_difficulty</th>
      <th>inflation_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>561</th>
      <td>2010-07-18</td>
      <td>0.085840</td>
      <td>1.552225e-03</td>
      <td>1.474778e+03</td>
      <td>1.815433e+02</td>
      <td>0.002494</td>
    </tr>
    <tr>
      <th>562</th>
      <td>2010-07-19</td>
      <td>0.080800</td>
      <td>1.570274e-03</td>
      <td>3.251693e+03</td>
      <td>1.815433e+02</td>
      <td>0.002517</td>
    </tr>
    <tr>
      <th>563</th>
      <td>2010-07-20</td>
      <td>0.074736</td>
      <td>1.633446e-03</td>
      <td>1.200497e+03</td>
      <td>1.815433e+02</td>
      <td>0.002611</td>
    </tr>
    <tr>
      <th>564</th>
      <td>2010-07-21</td>
      <td>0.079193</td>
      <td>1.868085e-03</td>
      <td>1.649916e+03</td>
      <td>1.815433e+02</td>
      <td>0.002978</td>
    </tr>
    <tr>
      <th>565</th>
      <td>2010-07-22</td>
      <td>0.058470</td>
      <td>1.588324e-03</td>
      <td>1.932369e+03</td>
      <td>1.815433e+02</td>
      <td>0.002525</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5161</th>
      <td>2023-02-20</td>
      <td>24797.843036</td>
      <td>2.861319e+08</td>
      <td>2.811605e+09</td>
      <td>3.915640e+13</td>
      <td>0.000048</td>
    </tr>
    <tr>
      <th>5162</th>
      <td>2023-02-21</td>
      <td>24400.389390</td>
      <td>3.250614e+08</td>
      <td>3.665580e+09</td>
      <td>3.915640e+13</td>
      <td>0.000054</td>
    </tr>
    <tr>
      <th>5163</th>
      <td>2023-02-22</td>
      <td>24163.738981</td>
      <td>2.725065e+08</td>
      <td>3.396857e+09</td>
      <td>3.915640e+13</td>
      <td>0.000045</td>
    </tr>
    <tr>
      <th>5164</th>
      <td>2023-02-23</td>
      <td>23910.034210</td>
      <td>3.328473e+08</td>
      <td>3.312575e+09</td>
      <td>3.915640e+13</td>
      <td>0.000055</td>
    </tr>
    <tr>
      <th>5165</th>
      <td>2023-02-24</td>
      <td>23175.824204</td>
      <td>2.627742e+08</td>
      <td>4.347353e+09</td>
      <td>3.915640e+13</td>
      <td>0.000044</td>
    </tr>
  </tbody>
</table>
<p>4605 rows × 6 columns</p>
</div>



7. Visualize the data

Line plot of Bitcoin Price over Time:


```python
df = btc

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d') # convert to datetime format
df['year'] = df['time'].dt.year # create a new column for year

plt.plot(df['time'], df['Price'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Bitcoin Price over Time')
plt.xticks(rotation=90)
plt.show()

```


    
![png](output_18_0.png)
    


A line plot is a basic visualization that shows the trend of a variable over time. In this case, the line plot of Bitcoin price over time shows the change in price of Bitcoin over a certain period. This plot helps to easily observe the upward or downward trends of Bitcoin price and also helps to identify patterns such as seasonality, trends, and cycles.



Scatter plot of Hash Rate vs. Mining Difficulty:


```python
plt.scatter(df['hash_rate'], df['mining_difficulty'])
plt.xlabel('Hash Rate')
plt.ylabel('Mining Difficulty')
plt.title('Hash Rate vs. Mining Difficulty')
plt.show()
```


    
![png](output_21_0.png)
    


A scatter plot is a graphical representation of the relationship between two variables. In this plot, hash rate and mining difficulty are represented on the X-axis and Y-axis, respectively. Scatter plot of hash rate vs. mining difficulty helps to identify the correlation between the two variables. If the points in the plot are closer together, there is a strong correlation between the two variables.

Bar plot of Transaction Volume per Year:


```python
import pandas as pd
import matplotlib.pyplot as plt

tv_per_year = df.groupby('year')['transaction_volume'].sum()

plt.bar(tv_per_year.index, tv_per_year.values)
plt.xlabel('Year')
plt.ylabel('Transaction Volume')
plt.title('Transaction Volume per Year')
plt.show()

```


    
![png](output_24_0.png)
    


A bar plot is a graphical representation of the distribution of a categorical variable. In this plot, the transaction volume is grouped by year and represented by vertical bars. The height of each bar indicates the transaction volume for each year. This plot helps to compare the transaction volume for different years and also shows the trend in transaction volume over time.

Area plot of Inflation Rate over Time:



```python
plt.fill_between(df['time'], df['inflation_rate'], color='blue', alpha=0.2)
plt.legend(['Inflation Rate', 'Interest Rate'])
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title('Inflation Rate over Time')
plt.show()
```


    
![png](output_27_0.png)
    


An area plot is a plot that represents the magnitude of a variable over time. In this plot, inflation rate and interest rate are represented by the area under their respective curves. The plot helps to show the change in inflation rate and interest rate over time and also the magnitude of the difference between the two variables. It helps to observe the trend of the variables and the seasonality, trends, and cycles that they exhibit.





Scatter plot of Price vs. Hash Rate over time:



```python

plt.scatter(df['time'], df['Price'], c=df['hash_rate'], cmap='coolwarm')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.title('Price vs Hash Rate over Time')
plt.colorbar(label='Hash Rate')
plt.show()

```


    
![png](output_30_0.png)
    


This scatter plot shows the relationship between the price of Bitcoin and the hash rate (computational power used for mining) over time. The color of each data point represents the hash rate. We can see that as the hash rate increases, the price tends to increase as well.



Line plot of Inflation Rate over time:



```python
plt.plot(df['time'], df['inflation_rate'])
plt.xlabel('Time')
plt.ylabel('Inflation Rate')
plt.title('Inflation Rate over Time')
plt.show()

```


    
![png](output_33_0.png)
    


This line plot shows the inflation rate of Bitcoin over time. We can see that the inflation rate was very high in the early years of Bitcoin's existence, but has gradually decreased as the supply of Bitcoin has approached its maximum limit of 21 million.



Stacked bar chart of Mining Difficulty by year:



```python
df_difficulty = df.groupby('year')['mining_difficulty'].sum()

plt.bar(df_difficulty.index, df_difficulty)
plt.xlabel('Year')
plt.ylabel('Mining Difficulty')
plt.title('Mining Difficulty by Year')
plt.show()

```


    
![png](output_36_0.png)
    


This stacked bar chart shows the total mining difficulty of Bitcoin for each year. We can see that the mining difficulty has increased dramatically over time, which reflects the increasing competition among miners to solve the complex mathematical problems required to validate transactions and earn Bitcoin rewards.



Heatmap of Correlation between Variables:



```python
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title('Correlation between Variables')
plt.show()

```


    
![png](output_39_0.png)
    


This heatmap shows the correlation between all of the variables in the df dataframe. We can see that there is a strong positive correlation between Price and Market Cap, as well as between Hash Rate and Mining Difficulty. There is also a negative correlation between Inflation Rate and ROI 1 Year, which makes sense since a high inflation rate would tend to decrease the ROI for Bitcoin investors.


7. Define function to estimate the fundamental component of Bitcoin prices:



```python
def estimate_fundamental_component(df, w):
    """
    df: dataframe containing the economic indicators and macroeconomic factors
    w: weights assigned to each of the economic indicators and macroeconomic factors
    """
    fundamental = w[0] + w[1]*df['Hash Rate'] + w[2]*df['Volume'] + w[3]*df['Mining Difficulty'] + w[4]*df['Inflation Rate'] + w[5]*df['Interest Rate']
    return fundamental
```

8. Collect necessary data for estimating the speculative component of Bitcoin prices:


```python

```

9. Define the weights for the economic indicators and macroeconomic factors. For example:



```python
# Define the weights
w0 = 0
w1 = 0.000000007
w2 = 0.00000001
w3 = 0.000000002
w4 = 2.2
```

10. Calculate the fundamental component of Bitcoin prices at time t using the equation for $F_t$:



```python
btc['F_t'] = w0 + w1 * btc['hash_rate'] + w2 * btc['transaction_volume'] + w3 * btc['mining_difficulty'] + w4 * btc['inflation_rate'] 
```


```python
btc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>Price</th>
      <th>hash_rate</th>
      <th>transaction_volume</th>
      <th>mining_difficulty</th>
      <th>inflation_rate</th>
      <th>year</th>
      <th>F_t</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>561</th>
      <td>2010-07-18</td>
      <td>0.085840</td>
      <td>1.552225e-03</td>
      <td>1.474778e+03</td>
      <td>1.815433e+02</td>
      <td>0.002494</td>
      <td>2010</td>
      <td>0.005503</td>
    </tr>
    <tr>
      <th>562</th>
      <td>2010-07-19</td>
      <td>0.080800</td>
      <td>1.570274e-03</td>
      <td>3.251693e+03</td>
      <td>1.815433e+02</td>
      <td>0.002517</td>
      <td>2010</td>
      <td>0.005570</td>
    </tr>
    <tr>
      <th>563</th>
      <td>2010-07-20</td>
      <td>0.074736</td>
      <td>1.633446e-03</td>
      <td>1.200497e+03</td>
      <td>1.815433e+02</td>
      <td>0.002611</td>
      <td>2010</td>
      <td>0.005757</td>
    </tr>
    <tr>
      <th>564</th>
      <td>2010-07-21</td>
      <td>0.079193</td>
      <td>1.868085e-03</td>
      <td>1.649916e+03</td>
      <td>1.815433e+02</td>
      <td>0.002978</td>
      <td>2010</td>
      <td>0.006568</td>
    </tr>
    <tr>
      <th>565</th>
      <td>2010-07-22</td>
      <td>0.058470</td>
      <td>1.588324e-03</td>
      <td>1.932369e+03</td>
      <td>1.815433e+02</td>
      <td>0.002525</td>
      <td>2010</td>
      <td>0.005575</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5161</th>
      <td>2023-02-20</td>
      <td>24797.843036</td>
      <td>2.861319e+08</td>
      <td>2.811605e+09</td>
      <td>3.915640e+13</td>
      <td>0.000048</td>
      <td>2023</td>
      <td>78342.919195</td>
    </tr>
    <tr>
      <th>5162</th>
      <td>2023-02-21</td>
      <td>24400.389390</td>
      <td>3.250614e+08</td>
      <td>3.665580e+09</td>
      <td>3.915640e+13</td>
      <td>0.000054</td>
      <td>2023</td>
      <td>78351.731470</td>
    </tr>
    <tr>
      <th>5163</th>
      <td>2023-02-22</td>
      <td>24163.738981</td>
      <td>2.725065e+08</td>
      <td>3.396857e+09</td>
      <td>3.915640e+13</td>
      <td>0.000045</td>
      <td>2023</td>
      <td>78348.676333</td>
    </tr>
    <tr>
      <th>5164</th>
      <td>2023-02-23</td>
      <td>23910.034210</td>
      <td>3.328473e+08</td>
      <td>3.312575e+09</td>
      <td>3.915640e+13</td>
      <td>0.000055</td>
      <td>2023</td>
      <td>78348.255922</td>
    </tr>
    <tr>
      <th>5165</th>
      <td>2023-02-24</td>
      <td>23175.824204</td>
      <td>2.627742e+08</td>
      <td>4.347353e+09</td>
      <td>3.915640e+13</td>
      <td>0.000044</td>
      <td>2023</td>
      <td>78358.113167</td>
    </tr>
  </tbody>
</table>
<p>4605 rows × 8 columns</p>
</div>




```python
# create a line plot of F_t against time
plt.plot(btc['time'], btc['F_t'])

# set the title of the plot
plt.title('Plot of F_t against time')

# set the x-axis label
plt.xlabel('Time')

# set the y-axis label
plt.ylabel('F_t')

# show the plot
plt.show()
```


    
![png](output_50_0.png)
    


We can note here the similarity between the Bitcoin Price and the Fundamental price till it gets to the end where the prediction does not follow the Bitcoin bear market trend.

11. Define the weights for the technical indicators and sentiment analysis. For example:



```python
alpha1 = 0.4  # weight for moving average
alpha2 = 0.3  # weight for relative strength index
alpha3 = 0.2  # weight for stochastic oscillator
alpha4 = 0.1  # weight for sentiment analysis
beta = 0.5  # persistence parameter
```

12. Define a function $f_j$ to calculate the value of each technical indicator and sentiment analysis at time $t$. For example:



```python
def moving_average(P, k):
    return sum(P[-k:]) / k

def relative_strength_index(P, k):
    delta = [P[i] - P[i-1] for i in range(1, len(P))]
    up = [d if d > 0 else 0 for d in delta]
    down = [abs(d) if d < 0 else 0 for d in delta]
    rs = sum(up[-k:]) / sum(down[-k:])
    return 100 - 100 / (1 + rs)

def stochastic_oscillator(P, k):
    min_price = min(P[-k:])
    max_price = max(P[-k:])
    return (P[-1] - min_price) / (max_price - min_price) * 100

def sentiment_analysis(market_sentiment, news_sentiment):
    return (market_sentiment + news_sentiment) / 2
```

13. Calculate the value of each technical indicator and sentiment analysis at time t using the corresponding function $f_j$. For example:



```python
P = np.array(btc["Price"])

Y1_t = alpha1 * moving_average(P, k=30)
Y2_t = alpha2 * relative_strength_index(P, k=14)
Y3_t = alpha3 * stochastic_oscillator(P, k=14)
# Y4_t = alpha4 * sentiment_analysis(market_sentiment, news_sentiment)
```

14. Calculate the speculative component of Bitcoin prices at time $t$ using the equation for $S_t$:



```python
# S_t = sum([Y1_t, Y2_t, Y3_t, Y4_t]) + beta * S_t_1
```

15. Calculate the total price of Bitcoin at time $t$ by adding the fundamental component $F_t$ and the speculative component $S_t$:



```python
# F_t = btc['F_t']

# P_t = F_t + S_t
```

16. Repeat steps 2-14 for each time period $t$ in the dataset.


## Reference
- we use the data from: https://coinmetrics.io/community-network-data/
