import pandas as pd

def read_data(file_path):
    """
    Reads a CSV file and returns a pandas dataframe.
    """
    return pd.read_csv(file_path)


def clean_and_transform_data(btc_full_df, btc_google_trend_df):
    """
    Cleans and transforms the data for further processing.
    """
    time = btc_full_df['time']
    price = btc_full_df['PriceUSD']
    hash_rate = btc_full_df['HashRate']
    transaction_volume = btc_full_df['TxTfrValAdjUSD']
    mining_difficulty = btc_full_df['DiffMean']
    issuance = btc_full_df['IssTotNtv']
    supply = btc_full_df['SplyCur']
    inflation_rate = issuance / supply
    df = pd.DataFrame({
        'time': time,
        'Price': price,
        'hash_rate': hash_rate,
        'transaction_volume': transaction_volume,
        'mining_difficulty': mining_difficulty,
        'inflation_rate': inflation_rate,
    })
    btc = pd.DataFrame(df)
    btc = btc.dropna()
    btc['time'] = pd.to_datetime(btc['time'])
    btc_google_trend_df['time'] = pd.to_datetime(btc_google_trend_df['time'])
    btc_google_trend_df = btc_google_trend_df.set_index('time')
    btc_google_trend_df = btc_google_trend_df.resample('D').interpolate(method='linear')
    btc_new = pd.merge(btc, btc_google_trend_df, on='time', how='outer')
    btc = btc_new.dropna()
    return btc


def clean_data(df):
    """
    Cleans a dataframe by dropping rows with missing values.
    """
    return df.dropna()