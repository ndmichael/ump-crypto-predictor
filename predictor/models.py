from django.db import models
from django.utils import timezone

class CryptoPair(models.Model):
    symbol = models.CharField(max_length=10, unique=True) # "BTCUSDT"
    name = models.CharField(max_length=50) # "BTC/USDT"
    timeframe = models.CharField(max_length=5)      # e.g., 1m, 5m
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.symbol


class Prediction(models.Model):
    pair = models.ForeignKey(CryptoPair, on_delete=models.CASCADE)
    predicted_price = models.FloatField()
    volume = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.crypto_symbol.symbol} ({self.time_frame}): {self.predicted_price}"

