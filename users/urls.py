from django.contrib import admin
from django.urls import path, include
from .views import user_dashboard

urlpatterns = [
    path('dashboard/', user_dashboard, name="userdashboard"),
    path('redirect/dashboard/', user_dashboard, name="user_dashboard"),
]