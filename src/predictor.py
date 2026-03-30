import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_fetcher import get_data, get_latest_price
from src.feature_engineering import build_features, get_feature_columns
from src.model_trainer import load_model, model_exists, full_train_pipeline, evaluate_model, time_series_split
from src.backtester import run_backtest, compute_metrics, buy_and_hold, INITIAL_CAPITAL_TRY, INITIAL_CAPITAL_USD


# ---------------------------------------------------------------------------
# Sinyal Eşikleri
# ---------------------------------------------------------------------------
SIGNAL_THRESHOLD_BUY  = 0.55   # >= 0.55 → Al
SIGNAL_THRESHOLD_SELL = 0.45   # <= 0.45 → Sat
# Arada kalan → Bekle


def _get_signal_label(proba: float) -> dict:
    """
    Olasılık değerine göre sinyal etiketi ve renk bilgisi döner.

    Args:
        proba: Modelin 'Al' (1) olasılığı [0, 1]

    Returns:
        {
          "signal"    : "AL" | "SAT" | "BEKLE",
          "color"     : "#renk_kodu",
          "emoji"     : str,
          "confidence": float (0-100),
        }
    """
    confidence = round(abs(proba - 0.5) * 200, 2)   # 0–100 arası güven skoru

    if proba >= SIGNAL_THRESHOLD_BUY:
        return {
            "signal"    : "AL",
            "color"     : "#1976D2",   # Mavi
            "emoji"     : "🔵",
            "confidence": confidence,
        }
    elif proba <= SIGNAL_THRESHOLD_SELL:
        return {
            "signal"    : "SAT",
            "color"     : "#D32F2F",   # Kırmızı
            "emoji"     : "🔴",
            "confidence": confidence,
        }
    else:
        return {
            "signal"    : "BEKLE",
            "color"     : "#F57C00",   # Turuncu
            "emoji"     : "🟡",
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Veri + Feature Hazırlama
# ---------------------------------------------------------------------------
def prepare_data(
    asset: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Ham veriyi çekip feature engineering uygular.

    Returns:
        (raw_df, feature_df, feature_cols)
        - raw_df      : Ham OHLCV verisi
        - feature_df  : Feature'lar eklenmiş ve temizlenmiş DataFrame
        - feature_cols: Modele verilecek kolon isimleri
    """
    raw_df       = get_data(asset, start=start, end=end)
    feature_df   = build_features(raw_df)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    return raw_df, feature_df, feature_cols


# ---------------------------------------------------------------------------
# Model Yükleme / Otomatik Eğitim
# ---------------------------------------------------------------------------
def get_or_train_model(
    asset: str,
    feature_df: pd.DataFrame,
    feature_cols: list,
    model_type: str = "xgboost",
    force_retrain: bool = False,
):
    """
    Varsa kaydedilmiş modeli yükler, yoksa eğitip kaydeder.

    Args:
        asset        : "Altın", "USD/TRY" veya "BTC"
        feature_df   : build_features() çıktısı
        feature_cols : Model feature kolonları
        model_type   : "xgboost" veya "random_forest"
        force_retrain: True ise mevcut model olsa dahi yeniden eğitir

    Returns:
        Yüklenmiş veya eğitilmiş model nesnesi
    """
    if not force_retrain and model_exists(asset):
        model = load_model(asset)
    else:
        result = full_train_pipeline(
            df          = feature_df,
            feature_cols= feature_cols,
            asset       = asset,
            model_type  = model_type,
            optimize    = False,
        )
        model = result["model"]
    return model


# ---------------------------------------------------------------------------
# En İyi Modeli Otomatik Seç
# ---------------------------------------------------------------------------
def select_best_model(
    asset: str,
    feature_df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    XGBoost ve Random Forest modellerini eğitir (kaydetmeden), test accuracy'lerine
    göre daha başarılı olanı seçer ve yalnızca kazananı kaydeder.

    Returns:
        (best_model, best_model_type, accuracy_xgb, accuracy_rf)
    """
    from src.model_trainer import train_model, evaluate_model, save_model

    X_train, X_test, y_train, y_test = time_series_split(feature_df, feature_cols)

    trained = {}
    accuracies = {}
    for mt in ["xgboost", "random_forest"]:
        model = train_model(X_train, y_train, model_type=mt, optimize=False)
        metrics = evaluate_model(model, X_test, y_test)
        trained[mt]    = model
        accuracies[mt] = metrics["accuracy"]

    acc_xgb = accuracies["xgboost"]
    acc_rf  = accuracies["random_forest"]

    best_type  = "xgboost" if acc_xgb >= acc_rf else "random_forest"
    best_model = trained[best_type]

    # Sadece kazananı kaydet
    save_model(best_model, asset)
    print(f"[Auto] XGBoost acc={acc_xgb:.4f} | RF acc={acc_rf:.4f} → Kazanan: {best_type.upper()}")

    return best_model, best_type, acc_xgb, acc_rf


# ---------------------------------------------------------------------------
# Tek Tahmin (Dashboard / API için)
# ---------------------------------------------------------------------------
def predict_next_day(
    asset: str,
    start: str = "2015-01-01",
    model_type: str = "xgboost",
    force_retrain: bool = False,
) -> dict:
    """
    Seçilen varlık için yarınki yön tahmini yapar.

    Args:
        asset        : "Altın", "USD/TRY" veya "BTC"
        start        : Veri başlangıç tarihi
        model_type   : "xgboost" veya "random_forest"
        force_retrain: Modeli yeniden eğit

    Returns:
        {
          "asset"      : str,
          "date"       : str,          # Tahmin tarihi (yarın)
          "price"      : float,        # Güncel kapanış
          "try_gram"   : float | None, # Sadece altın
          "signal"     : str,          # "AL" | "SAT" | "BEKLE"
          "color"      : str,          # Hex renk
          "emoji"      : str,
          "confidence" : float,        # 0–100
          "proba_up"   : float,        # Yukarı olasılığı (0–1)
          "model_type" : str,
        }
    """
    # 1. Veri ve feature hazırla
    raw_df, feature_df, feature_cols = prepare_data(asset, start=start)

    # 2. Kaydedilmiş modeli yükle (run_full_backtest zaten en iyisini seçip kaydetmiş olmalı)
    if not force_retrain and model_exists(asset):
        model = load_model(asset)
    else:
        model, _, _, _ = select_best_model(asset, feature_df, feature_cols)

    # 3. Son satırı tahmin için kullan (en güncel veri noktası)
    X_latest = feature_df[feature_cols].iloc[[-1]]
    proba    = float(model.predict_proba(X_latest)[0][1])

    # 4. Sinyal belirle
    signal_info = _get_signal_label(proba)

    # 5. Güncel fiyat bilgisi
    price_info = get_latest_price(asset)

    return {
        "asset"      : asset,
        "date"       : price_info["date"],
        "price"      : price_info["price"],
        "try_gram"   : price_info.get("try_gram"),
        "signal"     : signal_info["signal"],
        "color"      : signal_info["color"],
        "emoji"      : signal_info["emoji"],
        "confidence" : signal_info["confidence"],
        "proba_up"   : round(proba, 4),
        "model_type" : model_type,
    }


# ---------------------------------------------------------------------------
# Çoklu Gün Tahmini (Gelecek N gün yön sinyali)
# ---------------------------------------------------------------------------
def predict_multi_day(
    asset: str,
    n_days: int = 5,
    start: str = "2015-01-01",
    model_type: str = "xgboost",
    force_retrain: bool = False,
) -> list[dict]:
    """
    Son N gün için geriye dönük tahmin yapar.
    (Gerçek gelecek tahminleri zaman serisi modelleri olmadan mümkün değildir;
     bu fonksiyon test setindeki son N günü simüle eder.)

    Returns:
        Her eleman bir günü temsil eden sözlük listesi:
        [{"date": ..., "close": ..., "signal": ..., "proba_up": ...}, ...]
    """
    raw_df, feature_df, feature_cols = prepare_data(asset, start=start)
    if not force_retrain and model_exists(asset):
        model = load_model(asset)
    else:
        model, _, _, _ = select_best_model(asset, feature_df, feature_cols)

    tail_df  = feature_df.tail(n_days).reset_index(drop=True)
    X_tail   = tail_df[feature_cols]
    probas   = model.predict_proba(X_tail)[:, 1]

    price_col = "Close_TRY_gram" if (asset == "Altın" and "Close_TRY_gram" in tail_df.columns) else "Close"

    results = []
    for i, (_, row) in enumerate(tail_df.iterrows()):
        proba       = float(probas[i])
        signal_info = _get_signal_label(proba)
        results.append({
            "date"    : str(row["Date"])[:10] if "Date" in row else str(i),
            "close"   : round(float(row[price_col]), 4) if price_col in row else None,
            "signal"  : signal_info["signal"],
            "emoji"   : signal_info["emoji"],
            "color"   : signal_info["color"],
            "proba_up": round(proba, 4),
            "confidence": signal_info["confidence"],
        })

    return results


# ---------------------------------------------------------------------------
# Tam Backtest Pipeline (Dashboard için)
# ---------------------------------------------------------------------------
def run_full_backtest(
    asset: str,
    start: str = "2015-01-01",
    model_type: str = "auto",
    force_retrain: bool = False,
) -> dict:
    """
    Veri çekme → eğitim → backtest → metrik hesaplama adımlarını birleştiren
    tam pipeline. Dashboard'un tek fonksiyon çağırması yeterli.

    model_type="auto" ise XGBoost ve RF karşılaştırılır, kazananı kullanılır.

    Returns:
        {
          "backtest_df"    : pd.DataFrame,
          "bh_df"          : pd.DataFrame,
          "metrics"        : dict,
          "feature_df"     : pd.DataFrame,
          "raw_df"         : pd.DataFrame,
          "feature_cols"   : list,
          "model"          : model nesnesi,
          "initial_capital": float,
        }
    """
    # 1. Veri
    raw_df, feature_df, feature_cols = prepare_data(asset, start=start)

    # 2. Model seç
    if model_type == "auto" or force_retrain:
        model, chosen_type, acc_xgb, acc_rf = select_best_model(asset, feature_df, feature_cols)
        print(f"[Auto] XGBoost acc={acc_xgb:.4f} | RF acc={acc_rf:.4f} → Seçilen: {chosen_type}")
    else:
        model = get_or_train_model(asset, feature_df, feature_cols, model_type, force_retrain)

    # 3. Tüm veri üzerinde tahmin yap (backtest için)
    X_all     = feature_df[feature_cols]
    all_preds = model.predict(X_all)

    # 4. Backtest
    initial_capital = INITIAL_CAPITAL_TRY if asset == "Altın" else INITIAL_CAPITAL_USD
    backtest_df     = run_backtest(feature_df, all_preds, asset=asset, initial_capital=initial_capital)
    bh_df           = buy_and_hold(feature_df, asset=asset, initial_capital=initial_capital)
    metrics         = compute_metrics(backtest_df, initial_capital=initial_capital)

    return {
        "backtest_df"    : backtest_df,
        "bh_df"          : bh_df,
        "metrics"        : metrics,
        "feature_df"     : feature_df,
        "raw_df"         : raw_df,
        "feature_cols"   : feature_cols,
        "model"          : model,
        "initial_capital": initial_capital,
    }


# ---------------------------------------------------------------------------
# Hızlı Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for asset in ["BTC", "USD/TRY", "Altın"]:
        print(f"\n{'='*55}")
        print(f"  Tahmin: {asset}")
        print(f"{'='*55}")

        result = predict_next_day(asset, start="2020-01-01", force_retrain=True)
        print(f"  Tarih      : {result['date']}")
        print(f"  Fiyat      : {result['price']}")
        if result["try_gram"]:
            print(f"  TRY/gram   : {result['try_gram']}")
        print(f"  Sinyal     : {result['emoji']} {result['signal']}")
        print(f"  Güven      : %{result['confidence']}")
        print(f"  Yukarı Olas: {result['proba_up']}")

        print(f"\n  Son 5 gün tahmini:")
        multi = predict_multi_day(asset, n_days=5, start="2020-01-01")
        for row in multi:
            print(f"    {row['date']} | {row['emoji']} {row['signal']:5s} | %{row['confidence']:.1f}")
