from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from tensorflow.keras.models import load_model
from predictor.forms import CryptoPredictionForm
from predictor.models import CryptoPair, Prediction
from predictor.ml.utils import fetch_candlestick_data
import os
import pickle
from decimal import Decimal

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
                defaults={'name': name, 'timeframe': time_frame}
            )

            # Prediction.objects.create(
            #         pair=crypto_pair,
            #         predicted_price=Decimal(100),
            #         volume=Decimal(100),
            #         user=request.user
            #     )

            # messages.success(
            #     request, f"{name} : {time_frame} has been sent for analysis."
            # )
            # return redirect("prediction_result")

            try:
                # Load pre-trained model
                model_path = f"{crypto_pair.symbol}_{time_frame}_model.h5"
                scaler_path = f"{crypto_pair.symbol}_{time_frame}_scaler.pkl"

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return render(request, 'prediction/error.html', {'error': 'Model or scaler not found.'})

                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                data = fetch_candlestick_data(crypto_pair.symbol, time_frame)
                latest_price = data['Close'].values[-60:].reshape(-1, 1)
                scaled_input = scaler.transform(latest_price)
                scaled_input = scaled_input.reshape(1, -1, 1)
                predicted_price = model.predict(scaled_input)
                prediction_value = scaler.inverse_transform(predicted_price)[0][0]

                predicted_price_decimal = Decimal(prediction_value)
                
                # Save prediction with user reference
                Prediction.objects.create(
                    pair=crypto_pair,
                    predicted_price=predicted_price_decimal,
                    volume=0,
                    user=request.user
                )
                messages.success(
                request, f"{name} : {time_frame} has been sent for analysis."
                )
                return redirect("prediction_result")
            except Exception as e:
                messages.error(
                request, f"there was an error, Please confirm pairs."
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
    