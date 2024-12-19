from django.contrib import admin
from django.urls import path, include
from .views import prediction_result, predictions

urlpatterns = [
    path('dashboard/results', prediction_result, name="prediction_result"),
    path('dashboard/predictions', predictions, name="predictions"),
]