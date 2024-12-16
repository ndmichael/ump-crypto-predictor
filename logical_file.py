# Project: AI-Powered Cryptocurrency Market Prediction

## **Project Structure**

crypto_predict/    # Root project folder
    |
    |-- core/       # Core settings and configurations
    |   |-- __init__.py
    |   |-- settings.py
    |   |-- urls.py
    |   |-- wsgi.py
    |   |-- asgi.py
    |
    |-- prediction/ # Main app for AI prediction and API integration
    |   |-- __init__.py
    |   |-- admin.py
    |   |-- apps.py
    |   |-- forms.py
    |   |-- models.py
    |   |-- urls.py
    |   |-- views.py
    |   |-- tasks.py    # For asynchronous data updates
    |   |-- utils.py    # Helper functions (e.g., API handling, feature engineering)
    |   |-- templates/  # Frontend templates
    |   |-- static/     # CSS/JS/Images
    |   |-- migrations/ # DB migrations

---

## **Django Setup**

### **Install Dependencies:**
```bash
pip install django djangorestframework tensorflow pandas numpy scikit-learn requests celery redis
```

---

## **Backend Implementation**

### **Settings Configuration (core/settings.py)**
```python
INSTALLED_APPS = [
    ...,
    'rest_framework',
    'prediction',
]

# Redis for Celery (asynchronous tasks)
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
```

---

### **Models (prediction/models.py)**
```python
from django.db import models

class CryptoSymbol(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.symbol

class Prediction(models.Model):
    crypto_symbol = models.ForeignKey(CryptoSymbol, on_delete=models.CASCADE)
    time_frame = models.CharField(max_length=10)  # e.g., 1d, 1w, 1M
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crypto_symbol.symbol} ({self.time_frame}): {self.predicted_price}"
```

---

### **Forms (prediction/forms.py)**
```python
from django import forms
from .models import CryptoSymbol

TIME_FRAMES = (
    ('1m', '1 Minute'), ('3m', '3 Minutes'), ('5m', '5 Minutes'),
    ('15m', '15 Minutes'), ('30m', '30 Minutes'), ('1h', '1 Hour'),
    ('6h', '6 Hours'), ('12h', '12 Hours'), ('1d', '1 Day'),
    ('1w', '1 Week'), ('1M', '1 Month'),
)

class PredictionForm(forms.Form):
    crypto_symbol = forms.ModelChoiceField(queryset=CryptoSymbol.objects.all(), label="Select Cryptocurrency")
    time_frame = forms.ChoiceField(choices=TIME_FRAMES, label="Select Time Frame")
```

---

### **Utility Functions (prediction/utils.py)**
```python
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
```

---

### **AI Model Training (prediction/tasks.py)**
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from .utils import fetch_candlestick_data

# Train an LSTM model
def train_model(symbol, interval):
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
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    return model, scaler
```

---

### **Views (prediction/views.py)**
```python
from django.shortcuts import render
from .forms import PredictionForm
from .tasks import train_model
from .utils import fetch_candlestick_data

def predict_price(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            symbol = form.cleaned_data['crypto_symbol']
            time_frame = form.cleaned_data['time_frame']

            # Train model and predict
            model, scaler = train_model(symbol.symbol, time_frame)
            data = fetch_candlestick_data(symbol.symbol, time_frame)
            latest_price = data['Close'].values[-60:].reshape(-1, 1)
            scaled_input = scaler.transform(latest_price)
            scaled_input = scaled_input.reshape(1, -1, 1)
            predicted_price = model.predict(scaled_input)
            prediction = scaler.inverse_transform(predicted_price)[0][0]

            return render(request, 'prediction/result.html', {'prediction': prediction})
    else:
        form = PredictionForm()

    return render(request, 'prediction/form.html', {'form': form})
```

---

## **Frontend Templates**.
