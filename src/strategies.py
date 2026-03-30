import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Strateji Sabitleri
# ---------------------------------------------------------------------------
SMA_SHORT_COL  = "MA_5"
SMA_LONG_COL   = "MA_20"
RSI_OVERSOLD   = 30.0
RSI_OVERBOUGHT = 70.0


# ---------------------------------------------------------------------------
# 1) SMA Crossover Stratejisi
# ---------------------------------------------------------------------------
def sma_crossover_signals(
    df: pd.DataFrame,
    short_col: str = SMA_SHORT_COL,
    long_col: str  = SMA_LONG_COL,
) -> np.ndarray:
    """
    SMA Crossover Stratejisi.
    Kısa MA > Uzun MA  → AL  (1)
    Kısa MA <= Uzun MA → SAT (0)

    Args:
        df        : build_features() çıktısı (MA kolonları mevcut olmalı)
        short_col : Kısa hareketli ortalama kolon adı (varsayılan: MA_5)
        long_col  : Uzun hareketli ortalama kolon adı (varsayılan: MA_20)

    Returns:
        0/1 numpy array (df ile aynı uzunlukta)
    """
    if short_col not in df.columns or long_col not in df.columns:
        raise ValueError(
            f"DataFrame'de '{short_col}' veya '{long_col}' kolonu bulunamadı. "
            f"build_features() çıktısı kullandığınızdan emin olun."
        )

    signals = np.where(df[short_col].values > df[long_col].values, 1, 0)
    return signals.astype(int)


# ---------------------------------------------------------------------------
# 2) RSI Stratejisi
# ---------------------------------------------------------------------------
def rsi_signals(
    df: pd.DataFrame,
    oversold: float   = RSI_OVERSOLD,
    overbought: float = RSI_OVERBOUGHT,
) -> np.ndarray:
    """
    RSI Stratejisi.
    RSI < oversold  → AL  (1)  — aşırı satım bölgesi, geri dönüş beklenir
    RSI > overbought → SAT (0) — aşırı alım bölgesi, düzeltme beklenir
    Arada              → Önceki sinyali koru (trend takibi)

    Args:
        df         : build_features() çıktısı (RSI_14 kolonu mevcut olmalı)
        oversold   : Alım eşiği (varsayılan: 30)
        overbought : Satım eşiği (varsayılan: 70)

    Returns:
        0/1 numpy array (df ile aynı uzunlukta)
    """
    if "RSI_14" not in df.columns:
        raise ValueError(
            "DataFrame'de 'RSI_14' kolonu bulunamadı. "
            "build_features() çıktısı kullandığınızdan emin olun."
        )

    rsi     = df["RSI_14"].values
    signals = np.zeros(len(df), dtype=int)
    current = 0  # başlangıç: nakit

    for i in range(len(rsi)):
        if rsi[i] < oversold:
            current = 1   # aşırı satım → AL
        elif rsi[i] > overbought:
            current = 0   # aşırı alım  → SAT
        # else: önceki sinyali koru
        signals[i] = current

    return signals


# ---------------------------------------------------------------------------
# 3) ML Stratejisi
# ---------------------------------------------------------------------------
def ml_signals(
    df: pd.DataFrame,
    model,
    feature_cols: list,
) -> np.ndarray:
    """
    ML Model Stratejisi.
    Eğitilmiş modelin tahminlerini sinyal olarak kullanır.

    Args:
        df           : build_features() çıktısı
        model        : Eğitilmiş sklearn/XGBoost modeli
        feature_cols : Modelin kullandığı feature kolon isimleri

    Returns:
        0/1 numpy array (df ile aynı uzunlukta)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature kolonları DataFrame'de bulunamadı: {missing}")

    X       = df[feature_cols]
    signals = model.predict(X).astype(int)
    return signals


# ---------------------------------------------------------------------------
# Dispatcher — tek giriş noktası
# ---------------------------------------------------------------------------
def get_signals(
    strategy: str,
    df: pd.DataFrame,
    model=None,
    feature_cols: Optional[list] = None,
) -> np.ndarray:
    """
    Strateji adına göre sinyal üretir.

    Args:
        strategy     : "ML Modeli" | "SMA Crossover" | "RSI Stratejisi"
        df           : build_features() çıktısı
        model        : ML modeli (sadece ML stratejisi için)
        feature_cols : Feature isimleri (sadece ML stratejisi için)

    Returns:
        0/1 numpy array
    """
    if strategy == "ML Modeli":
        if model is None or feature_cols is None:
            raise ValueError("ML stratejisi için 'model' ve 'feature_cols' gereklidir.")
        return ml_signals(df, model, feature_cols)

    elif strategy == "SMA Crossover":
        return sma_crossover_signals(df)

    elif strategy == "RSI Stratejisi":
        return rsi_signals(df)

    else:
        raise ValueError(
            f"Bilinmeyen strateji: '{strategy}'. "
            f"Geçerli seçenekler: 'ML Modeli', 'SMA Crossover', 'RSI Stratejisi'"
        )


# ---------------------------------------------------------------------------
# Strateji Açıklamaları (UI için)
# ---------------------------------------------------------------------------
STRATEGY_INFO = {
    "ML Modeli": {
        "icon"       : "🤖",
        "description": "XGBoost/Random Forest modeli ile yön tahmini.",
        "logic"      : "Model tahmin = 1 → AL, tahmin = 0 → SAT",
        "color"      : "#1976D2",
    },
    "SMA Crossover": {
        "icon"       : "📈",
        "description": "Kısa ve uzun hareketli ortalamaların kesişimine dayalı strateji.",
        "logic"      : "MA5 > MA20 → AL | MA5 ≤ MA20 → SAT",
        "color"      : "#2E7D32",
    },
    "RSI Stratejisi": {
        "icon"       : "📉",
        "description": "RSI göstergesinin aşırı alım/satım bölgelerine dayalı strateji.",
        "logic"      : "RSI < 30 → AL | RSI > 70 → SAT | Arada → Bekle",
        "color"      : "#F57C00",
    },
}


# ---------------------------------------------------------------------------
# Hızlı Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_fetcher import get_data
    from src.feature_engineering import build_features, get_feature_columns

    df_raw  = get_data("BTC", start="2023-01-01")
    df_feat = build_features(df_raw)

    print(f"Veri: {len(df_feat)} satır\n")

    sma_sig = sma_crossover_signals(df_feat)
    rsi_sig = rsi_signals(df_feat)

    print(f"SMA sinyalleri — AL: {sma_sig.sum()} | SAT: {(sma_sig==0).sum()}")
    print(f"RSI sinyalleri — AL: {rsi_sig.sum()} | SAT: {(rsi_sig==0).sum()}")
