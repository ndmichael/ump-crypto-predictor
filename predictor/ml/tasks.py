import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .utils import fetch_candlestick_data
import pickle


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
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Model and scaler already exist for {symbol} at {interval}. Skipping training.")
        return

    # Fetch data for the past 60 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=60)
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


# Train an LSTM model
""" def train_model(symbol, interval, base_path):
    # Create dynamic paths for model and scaler
    model_path = os.path.join(base_path, f"{symbol}_{interval}_model.keras")
    scaler_path = os.path.join(base_path, f"{symbol}_{interval}_scaler.pkl")

    data = fetch_candlestick_data(symbol, interval)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare training data
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='loss', verbose=1)

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping, model_checkpoint])

    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler

"""