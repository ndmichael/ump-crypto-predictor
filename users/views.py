from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from predictor.models import CryptoPair, Prediction

from django.conf import settings
from datetime import datetime, timedelta

from django.core.serializers.json import DjangoJSONEncoder
import json

import logging
logger = logging.getLogger(__name__)

 
@login_required
def user_dashboard(request):
    
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
    