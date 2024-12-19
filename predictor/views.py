from django.shortcuts import render
from .models import Prediction


def prediction_result(request):


    context = {
        "title": "prediction result"
    }
    return render(request, 'predictor/predictor_result.html', context)

def predictions(request):

    predictions = Prediction.objects.all().order_by("-timestamp")

    context = {
        "title": "predictions",
        "predictions": predictions,
    }
    return render(request, 'predictor/predictions.html', context)
