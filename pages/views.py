from django.shortcuts import render

def index(request):
    context = {
        'title': 'UMP - Predictor'
    }
    return render(request, "pages/index.html", context)
