from django.contrib import admin
from django.urls import path, include
from .views import prediction_result

urlpatterns = [
    path('dashboard/results', prediction_result, name="prediction_result"),
]