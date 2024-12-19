from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm , UsernameField
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Div, Row, BaseInput
from crispy_bootstrap5.bootstrap5 import FloatingField, Field



class CustomSubmit(BaseInput):
    input_type = "submit"
    field_classes = "btn btn-custom btn-lg"

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
)

# Define the time frames as a tuple of tuples
TIME_FRAMES = (
    ("", "----"),
    ("15m", "15 Minutes"),
    ("30m", "30 Minutes"), 
    ("1h", "1 Hour"),
    ("2h", "2 Hours"),
    ("4h", "2 Hours"),
    ("1d", "1 Day"),
    ("1w", "1 Week"),
)

# Add the time frame to the form
class CryptoPredictionForm(forms.Form):
    crypto_symbol = forms.ChoiceField(choices=CRYPTO_SYMBOLS, label="Cryptocurrency")
    base_symbol = forms.ChoiceField(choices=BASE_SYMBOLS, label="Base Currency")
    time_frame = forms.ChoiceField(choices=TIME_FRAMES, label="Time Frame")

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        # self.helper.disable_csrf = True
        self.helper.form_tag = False
        self.helper.layout = Layout(
            Row(
                FloatingField("crypto_symbol", wrapper_class='col-md-6'),
                FloatingField("base_symbol", wrapper_class='col-md-6'),
                FloatingField("time_frame", wrapper_class='col-12'),
                Div(
                    CustomSubmit('submit', 'PREDICT NOW', css_class='col-12 px-2')
                )    
            ),
        )