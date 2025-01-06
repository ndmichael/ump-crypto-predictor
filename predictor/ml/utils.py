from itertools import product
import os
import numpy as np
import pandas as pd

import requests
import pickle
from datetime import datetime, timedelta

# Define your symbols and time frames
CRYPTO_SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "SHIB", "TRX", "AVAX", "ATOM", "LINK"]
BASE_SYMBOLS = ["USDT", "USDC", "EUR", "JPY", "TRY"]
TIME_FRAMES = ["15m", "30m", "1h", "4h", "1d", "1w"]

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

# Fetch available symbols
def fetch_crypto_symbols():
    response = requests.get(f"{BINANCE_BASE_URL}/exchangeInfo")
    symbols = response.json().get("symbols", [])
    return [(s["baseAsset"], s["quoteAsset"]) for s in symbols if s["status"] == "TRADING"]

# Function to fetch candlestick data with dynamic date ranges
def fetch_candlestick_data(symbol, interval, start_time, end_time):
    limit = 1000
    df = pd.DataFrame()

    while start_time < end_time:
        url = f"{BINANCE_BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}&startTime={int(start_time.timestamp() * 1000)}"
        response = requests.get(url)
        data = response.json()

        if not data:
            print(f"No data returned for {symbol} at {interval}.")
            break

        temp_df = pd.DataFrame(data, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'])

        if temp_df.empty:
            print(f"No data available for {symbol} at {interval}.")
            break

        df = pd.concat([df, temp_df], ignore_index=True)
        start_time = datetime.fromtimestamp(temp_df['CloseTime'].iloc[-1] / 1000)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol} at {interval}.")

    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)



