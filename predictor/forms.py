from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm , UsernameField
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Div, Row, BaseInput
from crispy_bootstrap5.bootstrap5 import FloatingField, Field


CRYPTO_SYMBOLS = (
    ("BTC", "Bitcoin (BTC)"),
    ("ETH", "Ethereum (ETH)"),
    ("BNB", "Binance Coin (BNB)"),
    ("XRP", "Ripple (XRP)"),
    ("ADA", "Cardano (ADA)"),
    ("SOL", "Solana (SOL)"),
    ("DOGE", "Dogecoin (DOGE)"),
    ("DOT", "Polkadot (DOT)"),
    ("MATIC", "Polygon (MATIC)"),
    ("SHIB", "Shiba Inu (SHIB)"),
    ("LTC", "Litecoin (LTC)"),
    ("TRX", "TRON (TRX)"),
    ("AVAX", "Avalanche (AVAX)"),
    ("ATOM", "Cosmos (ATOM)"),
    ("LINK", "Chainlink (LINK)"),
)

BASE_SYMBOLS = (
    ("USDT", "Tether (USDT)"),
    ("BUSD", "Binance USD (BUSD)"),
    ("USDC", "USD Coin (USDC)"),
    ("BTC", "Bitcoin (BTC)"),
    ("ETH", "Ethereum (ETH)"),
    ("BNB", "Binance Coin (BNB)"),
    ("EUR", "Euro (EUR)"),
    ("GBP", "British Pound (GBP)"),
    ("AUD", "Australian Dollar (AUD)"),
    ("JPY", "Japanese Yen (JPY)"),
    ("TRY", "Turkish Lira (TRY)"),
    ("RUB", "Russian Ruble (RUB)"),
    ("CAD", "Canadian Dollar (CAD)"),
    ("CNY", "Chinese Yuan (CNY)"),
    ("KRW", "South Korean Won (KRW)"),
)

# Define the time frames as a tuple of tuples
TIME_FRAMES = (
    ("5m", "5 Minutes"),
    ("15m", "15 Minutes"),
    ("30m", "30 Minutes"),
    ("1h", "1 Hour"),
    ("12h", "12 Hours"),
    ("1d", "1 Day"),
    ("3d", "3 Days"),
    ("1w", "1 Week"),
    ("1M", "1 Month"),
)

# Add the time frame to the form
class CryptoPredictionForm(forms.Form):
    crypto_symbol = forms.ChoiceField(choices=CRYPTO_SYMBOLS, label="Cryptocurrency")
    base_symbol = forms.ChoiceField(choices=BASE_SYMBOLS, label="Base Currency")
    time_frame = forms.ChoiceField(choices=TIME_FRAMES, label="Time Frame")

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout(
            Field("status", wrapper_class='col-md-12 control-lg'),
            Field("comment",),
            Submit('submit', "Submit Payment", css_class="btn-lg")
        )