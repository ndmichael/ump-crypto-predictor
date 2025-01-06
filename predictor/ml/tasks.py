from celery import shared_task
from itertools import product
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .utils import fetch_candlestick_data
from datetime import datetime, timedelta
import pickle


# Define your symbols and time frames
CRYPTO_SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "SHIB", " ", "TRX", "AVAX", "ATOM", "LINK"]
BASE_SYMBOLS = ["USDT", "USDC", "EUR", "JPY", "TRY"]
TIME_FRAMES = ["15m", "30m", "1h", "4h", "1d", "1w"]

# Base path for saving models and scalers
base_path = os.getcwd()

# Function to prepare data
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

# Function to train the model
def train_model(symbol, interval, base_path):
    model_path = os.path.join(base_path, f"{symbol}_{interval}_model.keras")
    scaler_path = os.path.join(base_path, f"{symbol}_{interval}_scaler.pkl")

    # Check if model and scaler exist
    # if os.path.exists(model_path) and os.path.exists(scaler_path):
    #     print(f"Model and scaler already exist for {symbol} at {interval}. Skipping training.")
    #     return

    # Fetch data for the past 120 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=120)
    data = fetch_candlestick_data(symbol, interval, start_time, end_time)

    # Prepare data
    x_train, y_train, scaler = prepare_data(data)

    # Build GRU model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='loss', verbose=1)

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping, model_checkpoint])

    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model and scaler saved for {symbol} at {interval}.")


@shared_task
def train_model_task(crypto_symbol, base_symbol, time_frame):
    symbol = f"{crypto_symbol}{base_symbol}"
    print(f"Training model for {symbol} at {time_frame} interval")
    try:
        train_model(symbol, time_frame, base_path)
    except Exception as e:
        print(f"Failed to train model for {symbol} at {time_frame}: {e}")


@shared_task
def train_all_models():
    for crypto_symbol, base_symbol, time_frame in product(CRYPTO_SYMBOLS, BASE_SYMBOLS, TIME_FRAMES):
        train_model_task.delay(crypto_symbol, base_symbol, time_frame)