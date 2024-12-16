from django.shortcuts import render


def prediction_result(request):


    context = {
        "title": "prediction result"
    }
    return render(request, 'predictor/predictor_result.html', context)
