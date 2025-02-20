from django.shortcuts import render, redirect
from .models import Prediction
from django.contrib import messages
from django.contrib.auth.decorators import login_required


@login_required
def prediction_result(request, prediction_id):
    if not prediction_id:
        messages.error(request, "No prediction found.")
        return redirect("user_dashboard")

    try:
        prediction = Prediction.objects.get(id=prediction_id, user=request.user)
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found.")
        return redirect("user_dashboard")

    context = {
        "title": "prediction result",
        "prediction": prediction,
    }
    return render(request, 'predictor/predictor_result.html', context)

def predictions(request):

    predictions = Prediction.objects.all().order_by("-timestamp")

    context = {
        "title": "predictions",
        "predictions": predictions,
    }
    return render(request, 'predictor/predictions.html', context)


def user_settings(request):

    predictions = Prediction.objects.all().order_by("-timestamp")

    context = {
        "title": "settings",
    }
    return render(request, 'predictor/user_settings.html', context)
