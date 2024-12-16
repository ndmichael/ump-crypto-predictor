import requests
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# Fetch available symbols
def fetch_crypto_symbols():
    response = requests.get(f"{BINANCE_BASE_URL}/exchangeInfo")
    symbols = response.json().get("symbols", [])
    return [(s["baseAsset"], s["quoteAsset"]) for s in symbols if s["status"] == "TRADING"]

# Fetch candlestick data
def fetch_candlestick_data(symbol, interval, limit=100):
    url = f"{BINANCE_BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'])
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)