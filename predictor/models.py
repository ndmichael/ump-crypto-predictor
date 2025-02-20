from django.db import models
from django.utils import timezone
from users.models import CustomUser

class CryptoPair(models.Model):
    symbol = models.CharField(max_length=10) # "BTCUSDT"
    name = models.CharField(max_length=50) # "BTC/USDT"
    timeframe = models.CharField(max_length=5)      # e.g., 1m, 5m
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('symbol', 'timeframe')

    def __str__(self):
        return f"{self.symbol} {self.timeframe}"


class Prediction(models.Model):
    pair = models.ForeignKey(CryptoPair, on_delete=models.CASCADE)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, default=1)
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    current_price = models.DecimalField(max_digits=20, decimal_places=8, default=0.00)
    stop_loss = models.DecimalField(max_digits=20, decimal_places=8, default=0.00)
    take_profit = models.DecimalField(max_digits=20, decimal_places=8, default=0.00)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)
    signal = models.CharField(max_length=4, choices=[("BUY", "BUY"), ("SELL", "SELL"), ("HOLD", "HOLD")], default="HOLD")  # Buy/Sell Signal
    volume = models.DecimalField(max_digits=10, decimal_places=2)
    timestamp = models.DateTimeField(default=timezone.now)


    def __str__(self):
        return f"{self.pair.symbol} ({self.pair.timeframe}): {self.predicted_price} by {self.user.username}"

