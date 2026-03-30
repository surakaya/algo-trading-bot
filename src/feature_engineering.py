import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """MA 5, 20, 50 günlük hareketli ortalamalar."""
    df = df.copy()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """1 ve 3 günlük gecikmeli fiyatlar."""
    df = df.copy()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_3"] = df["Close"].shift(3)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI (Relative Strength Index) hesapla — pandas EWM ile Wilder smoothing."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing = EWM com=(period-1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def add_momentum(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Momentum = Close(t) / Close(t - period)"""
    df = df.copy()
    df["Momentum"] = df["Close"] / df["Close"].shift(period)
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: orta, üst ve alt bantlar."""
    df = df.copy()
    rolling_mean = df["Close"].rolling(window=period).mean()
    rolling_std = df["Close"].rolling(window=period).std()
    df["BB_Mid"] = rolling_mean
    df["BB_Upper"] = rolling_mean + (std_dev * rolling_std)
    df["BB_Lower"] = rolling_mean - (std_dev * rolling_std)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / df["BB_Width"].replace(0, np.nan)
    return df


def add_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Günlük getiri bazlı volatilite (yıllıklaştırılmış)."""
    df = df.copy()
    daily_return = df["Close"].pct_change()
    df["Volatility"] = daily_return.rolling(window=period).std() * np.sqrt(252)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hedef değişken: yarınki kapanış bugünkünden yüksekse 1 (Al), değilse 0 (Sat).
    direction = 1 → fiyat çıkıyor
    direction = 0 → fiyat düşüyor
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def build_features(df: pd.DataFrame, include_bollinger: bool = True, include_volatility: bool = True) -> pd.DataFrame:
    """
    Tüm feature engineering adımlarını sırasıyla uygular.
    Sonunda NaN içeren satırları düşürür.

    Args:
        df: Ham OHLCV verisi (Open, High, Low, Close, Volume kolonları olmalı)
        include_bollinger: Bollinger Bands eklensin mi
        include_volatility: Volatilite eklensin mi

    Returns:
        Feature'ları eklenmiş ve temizlenmiş DataFrame
    """
    df = df.copy()

    # Temel feature'lar
    df = add_moving_averages(df)
    df = add_lag_features(df)
    df = add_rsi(df)
    df = add_momentum(df)

    # Opsiyonel feature'lar
    if include_bollinger:
        df = add_bollinger_bands(df)
    if include_volatility:
        df = add_volatility(df)

    # Hedef değişken
    df = add_target(df)

    # NaN satırları düşür
    df = df.dropna()
    df = df.reset_index(drop=False)  # Date index'i kolon olarak sakla

    return df


def get_feature_columns(include_bollinger: bool = True, include_volatility: bool = True) -> list:
    """
    Model eğitiminde kullanılacak feature kolon isimlerini döner.
    build_features() ile tutarlı olmalı.
    """
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_20", "MA_50",
        "Lag_1", "Lag_3",
        "RSI_14",
        "Momentum",
    ]
    if include_bollinger:
        features += ["BB_Mid", "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct"]
    if include_volatility:
        features += ["Volatility"]

    return features
