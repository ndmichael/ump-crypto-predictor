from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np

from predictor.forms import CryptoPredictionForm
from predictor.models import CryptoPair, Prediction
from predictor.ml.utils import fetch_candlestick_data, mc_dropout_prediction, calculate_confidence

import os
import pickle
from decimal import Decimal
from django.conf import settings
from datetime import datetime, timedelta

from django.core.serializers.json import DjangoJSONEncoder
import json

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
                # predicted_price = model.predict(scaled_input)
                # prediction_value = scaler.inverse_transform(predicted_price)[0][0]

                # Convert predicted price to Decimal
                # predicted_price_decimal = Decimal(str(float(prediction_value)))

                # Instead of a single prediction, use MC dropout to get multiple predictions
                mean_pred, std_pred = mc_dropout_prediction(model, scaled_input, n_iter=50)
                # Inverse transform the mean prediction to get the price value
                prediction_value = scaler.inverse_transform(mean_pred)[0][0]
                predicted_price_decimal = Decimal(str(float(prediction_value)))

                # Calculate confidence using the relative uncertainty
                confidence = calculate_confidence(mean_pred, std_pred)[0][0]
                # Round confidence to two decimal places
                confidence = round(confidence, 2)
                confidence_decimal = Decimal(str(confidence))


                # Get latest actual price
                current_price = Decimal(str(float(data['Close'].iloc[-1])))

                # Calculate market volatility using ATR
                atr = Decimal(str(float(data['High'].iloc[-10:].mean() - data['Low'].iloc[-10:].mean())))  
                
                # Adjust Stop-Loss & Take-Profit dynamically based on ATR
                stop_loss = predicted_price_decimal - (atr * Decimal('1.5'))  
                take_profit = predicted_price_decimal + (atr * Decimal('2'))  

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
                    signal=signal,
                    confidence_score=confidence_decimal,
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

    
    # Get all user predictions
    user_predictions = Prediction.objects.filter(user=request.user)

    # Total predictions count
    total_predictions = user_predictions.count()

    # Last prediction (most recent)
    last_prediction = user_predictions.order_by('-timestamp').first()

    
    # Model "accuracy" (Measure relative difference between predicted & current price)
    accuracy = None
    if user_predictions.exists():
        diffs = []
        for p in user_predictions.exclude(current_price=0):
            diff = abs(p.predicted_price - p.current_price) / p.current_price * 100
            diffs.append(diff)
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            accuracy = round(100 - avg_diff, 2)  # e.g. 92.45% accurate
    
    # Determine next forecast direction (based on most recent signal)
    next_forecast = None
    if last_prediction:
        if last_prediction.signal == "BUY":
            next_forecast = "Likely Bullish - Consider holding or buying dips"
        elif last_prediction.signal == "SELL":
            next_forecast = "Possible Bearish - Watch for breakdown"
        else:
            next_forecast = "Neutral - Market indecisive"

    predictions = list(Prediction.objects.filter(user=request.user).order_by('-timestamp')[:20])
    predictions.reverse()  # ensures chronological order

    if len(predictions) < 2:
        labels = [datetime.now().strftime('%b %d')]
        data = [0]
    else:
        labels = [p.timestamp.strftime('%b %d') for p in predictions]
        data = [float(p.predicted_price) for p in predictions]

    last_updated = predictions[-1].timestamp.strftime('%b %d, %Y %I:%M %p') if predictions else "No Data"
    current_pair = predictions[-1].pair.name if predictions else "N/A"

    ai_sentiment = "bullish" if last_prediction and last_prediction.signal == "BUY" else \
               "bearish" if last_prediction and last_prediction.signal == "SELL" else "neutral"

    context = {
        "title": "User Dashboard",
        "crypto_form": form,
        "total_predictions": total_predictions,
        "last_prediction": last_prediction,
        "accuracy": accuracy,
        "next_forecast": next_forecast,
        "chart_labels": json.dumps(labels, cls=DjangoJSONEncoder),
        "chart_data": json.dumps(data, cls=DjangoJSONEncoder),
        "chart_pair": current_pair,
        "last_updated": last_updated,
        "ai_sentiment": ai_sentiment.capitalize()
    }
    return render(request, "users/user_dashboard.html", context)


@login_required
def redirect_dashboard(request):
    return redirect('user_dashboard')
    # if request.user.is_staff:
    #     return redirect('admindashboard')
    