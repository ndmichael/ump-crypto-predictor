from django.contrib import admin
from .models import CryptoPair, Prediction

admin.site.register(CryptoPair)
# admin.site.register(Prediction)

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('pair', 'user', 'predicted_price', 'volume', 'timestamp')
