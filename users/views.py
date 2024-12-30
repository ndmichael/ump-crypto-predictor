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
from django.conf import settings

 
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

                data = fetch_candlestick_data(crypto_pair.symbol, time_frame)
                latest_price = data['Close'].values[-60:].reshape(-1, 1)
                scaled_input = scaler.transform(latest_price)
                scaled_input = scaled_input.reshape(1, -1, 1)
                predicted_price = model.predict(scaled_input)
                prediction_value = scaler.inverse_transform(predicted_price)[0][0]

                # Convert predicted price to Decimal
                predicted_price_decimal = Decimal(str(float(prediction_value)))

                # Convert volume to Decimal
                volume_value = data["Volume"].iloc[-1]  # Use the last volume value
                volume_decimal = Decimal(str(float(volume_value)))

                
                # Save prediction with user reference
                Prediction.objects.create(
                    pair=crypto_pair,
                    predicted_price=predicted_price_decimal,
                    volume=volume_decimal,
                    user=request.user
                )
                messages.success(
                request, f"{name} : {time_frame} has been sent for analysis."
                )
                return redirect("prediction_result")
            except Exception as e:
                print("Error: ", e)
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
    