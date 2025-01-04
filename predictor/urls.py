from django.contrib import admin
from django.urls import path, include
from .views import prediction_result, predictions, user_settings

urlpatterns = [
    path('dashboard/result/<int:prediction_id>/', prediction_result, name="prediction_result"),
    path('dashboard/predictions', predictions, name="predictions"),
    path('dashboard/profile/settings', user_settings, name="user_settings"),
]