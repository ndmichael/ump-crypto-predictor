from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np

from predictor.forms import CryptoPredictionForm
from predictor.models import CryptoPair, Prediction
from predictor.ml.utils import fetch_candlestick_data

import os
import pickle
from decimal import Decimal
from django.conf import settings
from datetime import datetime, timedelta

import logging
logger = logging.getLogger(__name__)

 
@login_required
def user_dashboard(request):
    if request.method == 'POST':
        form = CryptoPredictionForm(request.POST)
        if form.is_valid():
            crypto_symbol = form.cleaned_data['crypto_symbol']
            base_symbol = form.cleaned_data['base_symbol']
            time_frame = form.cleaned_data['time_frame']

            symbol = str(crypto_symbol + base_symbol)
            name = str(crypto_symbol + "/" + base_symbol)

            # Create or retrieve the CryptoPair instance
            crypto_pair, created = CryptoPair.objects.get_or_create(
                symbol=symbol,
                timeframe= time_frame,
                defaults={'name': name}
            )

            try:
                # Load pre-trained model
                # Construct the file paths for the model and scaler
                model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ai_models', f"{crypto_pair.symbol}_{time_frame}_model.keras")
                scaler_path = os.path.join(settings.BASE_DIR, 'predictor', 'ai_models', f"{crypto_pair.symbol}_{time_frame}_scaler.pkl")

                print(f"scaler: ${scaler_path}")
                print(f"model: ${model_path}")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    # return render(request, 'prediction/error.html', {'error': 'Model or scaler not found.'})
                    messages.error(
                        request, f"Please confirm pairs & intervals"
                    )
                    return redirect("user_dashboard")

                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                # Determine the time range for fetching data
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=5)

                data = fetch_candlestick_data(crypto_pair.symbol, time_frame, start_time, end_time)
                latest_prices = data['Close'].values[-60:].reshape(-1, 1)
                
                scaled_input = scaler.transform(latest_prices)
                scaled_input = scaled_input.reshape(1, -1, 1)

                # Predict price
                predicted_price = model.predict(scaled_input)
                prediction_value = scaler.inverse_transform(predicted_price)[0][0]

                # Convert predicted price to Decimal
                predicted_price_decimal = Decimal(str(float(prediction_value)))

                # Get latest actual price
                current_price = Decimal(str(float(data['Close'].iloc[-1])))

                # Calculate market volatility using ATR
                atr = Decimal(str(float(data['High'].iloc[-10:].mean() - data['Low'].iloc[-10:].mean())))  
                
                # Adjust Stop-Loss & Take-Profit dynamically based on ATR
                stop_loss = predicted_price_decimal - (atr * Decimal('1.5'))  
                take_profit = predicted_price_decimal + (atr * Decimal('2'))  

                # Calculate confidence as the percentage difference between the predicted and current price.
                # You can modify this logic if you have a model-based confidence measure.
                confidence_score = abs((predicted_price_decimal - current_price) / current_price) * Decimal('100')

                # Determine Buy/Sell Signal
                if predicted_price_decimal > current_price * Decimal('1.01'):
                    signal = "BUY"
                elif predicted_price_decimal < current_price * Decimal('0.99'):
                    signal = "SELL"
                else:
                    signal = "HOLD"

                # Convert volume to Decimal
                volume_value = data["Volume"].iloc[-1]  # Use the last volume value
                volume_decimal = Decimal(str(float(volume_value)))

                
                # Save prediction with user reference
                prediction = Prediction.objects.create(
                    pair=crypto_pair,
                    predicted_price=predicted_price_decimal,
                    volume=volume_decimal,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence_score = confidence_score,
                    signal=signal,
                    user=request.user
                )
                messages.success(
                    request, f"{name} : {time_frame} Analysis Completed."
                )
                return redirect(f"prediction_result", prediction_id=prediction.id)
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                messages.error(
                    request, "An unexpected error occurred. Check pairs Please try again later."
                )
                return redirect("user_dashboard")
    else:
        form = CryptoPredictionForm()

    context = {
        "title": "User Dashboard",
        "crypto_form": form
    }
    return render(request, "users/user_dashboard.html", context)


@login_required
def redirect_dashboard(request):
    return redirect('user_dashboard')
    # if request.user.is_staff:
    #     return redirect('admindashboard')
    