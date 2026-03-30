from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

# src modüllerine erişim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_fetcher import get_data, get_latest_price
from src.feature_engineering import build_features, get_feature_columns
from src.model_trainer import (
    full_train_pipeline,
    load_model,
    model_exists,
    evaluate_model,
    get_feature_importance,
    time_series_split,
)
from src.backtester import run_backtest, compute_metrics, buy_and_hold, INITIAL_CAPITAL_USD, INITIAL_CAPITAL_TRY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Algo Trading Bot API",
    description="Altın / USD/TRY / BTC için ML tabanlı yön tahmin ve backtesting API'si",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------
VALID_ASSETS = ["Altın", "USD/TRY", "BTC"]
VALID_MODELS = ["xgboost", "random_forest"]


def validate_asset(asset: str):
    if asset not in VALID_ASSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz varlık: '{asset}'. Geçerli seçenekler: {VALID_ASSETS}",
        )


def validate_model_type(model_type: str):
    if model_type not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz model türü: '{model_type}'. Geçerli seçenekler: {VALID_MODELS}",
        )


# ---------------------------------------------------------------------------
# Şemalar
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    asset: str = "BTC"
    model_type: str = "xgboost"
    start: str = "2018-01-01"
    end: Optional[str] = None
    train_ratio: float = 0.80
    optimize: bool = False


class PredictResponse(BaseModel):
    ticker: str
    asset: str
    prediction: str          # "up" / "down"
    signal: str              # "Al" / "Sat"
    confidence: float        # 0.0 – 1.0
    current_price: float
    try_gram_price: Optional[float]
    date: str


class TrainResponse(BaseModel):
    asset: str
    model_type: str
    accuracy: float
    directional_accuracy: float
    train_rows: int
    test_rows: int
    message: str


class BacktestResponse(BaseModel):
    asset: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_buys: int
    num_sells: int
    win_rate_pct: float
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """API sağlık kontrolü."""
    return {
        "status": "ok",
        "message": "Algo Trading Bot API çalışıyor 🚀",
        "endpoints": ["/predict", "/train", "/backtest", "/latest-price", "/feature-importance"],
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# /latest-price
# ---------------------------------------------------------------------------
@app.get("/latest-price", tags=["Data"])
def latest_price(
    asset: str = Query("BTC", description="Altın | USD/TRY | BTC"),
):
    """Seçilen varlığın en güncel fiyat bilgisini döner."""
    validate_asset(asset)
    try:
        info = get_latest_price(asset)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
@app.get("/predict", tags=["Prediction"], response_model=PredictResponse)
def predict(
    asset: str = Query("BTC", description="Altın | USD/TRY | BTC"),
    start: str = Query("2018-01-01", description="Eğitim verisi başlangıç tarihi"),
    end: Optional[str] = Query(None, description="Eğitim verisi bitiş tarihi"),
):
    """
    Seçilen varlık için bir sonraki günün yön tahmini yapar.

    - Model daha önce eğitilmişse direkt yükler.
    - Eğitilmemişse önce eğitir, sonra tahmin yapar.

    Returns:
        ticker, prediction (up/down), confidence, current_price, ...
    """
    validate_asset(asset)

    try:
        # Veri çek
        raw_df = get_data(asset, start=start, end=end)
        feat_df = build_features(raw_df)
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in feat_df.columns]

        # Model yükle veya eğit
        if model_exists(asset):
            model = load_model(asset)
        else:
            result = full_train_pipeline(
                df=feat_df,
                feature_cols=feature_cols,
                asset=asset,
                model_type="xgboost",
                optimize=False,
            )
            model = result["model"]

        # Son satır üzerinden tahmin yap
        last_row = feat_df[feature_cols].iloc[[-1]]
        pred = int(model.predict(last_row)[0])
        proba = float(model.predict_proba(last_row)[0][1])

        # Güncel fiyat bilgisi
        price_info = get_latest_price(asset)

        from src.data_fetcher import TICKERS
        ticker_symbol = TICKERS.get(asset, asset)

        return PredictResponse(
            ticker=ticker_symbol,
            asset=asset,
            prediction="up" if pred == 1 else "down",
            signal="Al" if pred == 1 else "Sat",
            confidence=round(proba if pred == 1 else 1 - proba, 4),
            current_price=price_info["price"],
            try_gram_price=price_info.get("try_gram"),
            date=price_info["date"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /train
# ---------------------------------------------------------------------------
@app.post("/train", tags=["Training"], response_model=TrainResponse)
def train(req: TrainRequest):
    """
    Seçilen varlık için modeli sıfırdan eğitir ve kaydeder.

    - model_type: "xgboost" veya "random_forest"
    - optimize: True ise GridSearchCV uygular (yavaş!)
    """
    validate_asset(req.asset)
    validate_model_type(req.model_type)

    try:
        raw_df = get_data(req.asset, start=req.start, end=req.end)
        feat_df = build_features(raw_df)
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in feat_df.columns]

        result = full_train_pipeline(
            df=feat_df,
            feature_cols=feature_cols,
            asset=req.asset,
            model_type=req.model_type,
            train_ratio=req.train_ratio,
            optimize=req.optimize,
        )

        metrics = result["metrics"]
        split_idx = int(len(feat_df) * req.train_ratio)

        return TrainResponse(
            asset=req.asset,
            model_type=req.model_type,
            accuracy=metrics["accuracy"],
            directional_accuracy=metrics["directional_accuracy"],
            train_rows=split_idx,
            test_rows=len(feat_df) - split_idx,
            message=f"Model başarıyla eğitildi ve kaydedildi. Accuracy: {metrics['accuracy']:.4f}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /backtest
# ---------------------------------------------------------------------------
@app.get("/backtest", tags=["Backtest"], response_model=BacktestResponse)
def backtest(
    asset: str = Query("BTC", description="Altın | USD/TRY | BTC"),
    start: str = Query("2018-01-01", description="Başlangıç tarihi"),
    end: Optional[str] = Query(None, description="Bitiş tarihi"),
    initial_capital: Optional[float] = Query(None, description="Başlangıç sermayesi"),
):
    """
    Seçilen varlık için backtesting simülasyonu çalıştırır.

    - Eğitilmiş modelin test seti üzerinde backtest yapar.
    - Model yoksa önce eğitir.
    """
    validate_asset(asset)

    try:
        raw_df = get_data(asset, start=start, end=end)
        feat_df = build_features(raw_df)
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in feat_df.columns]

        # Model yükle veya eğit
        if model_exists(asset):
            model = load_model(asset)
        else:
            result = full_train_pipeline(
                df=feat_df,
                feature_cols=feature_cols,
                asset=asset,
                model_type="xgboost",
                optimize=False,
            )
            model = result["model"]

        # Test seti üzerinde backtest
        _, X_test, _, _ = time_series_split(feat_df, feature_cols)
        split_idx = int(len(feat_df) * 0.80)
        df_test = feat_df.iloc[split_idx:].copy()

        preds = model.predict(X_test)

        cap = initial_capital or (INITIAL_CAPITAL_TRY if asset == "Altın" else INITIAL_CAPITAL_USD)

        bt_df = run_backtest(df_test, preds, asset=asset, initial_capital=cap)
        metrics = compute_metrics(bt_df, initial_capital=cap)

        return BacktestResponse(
            asset=asset,
            **metrics,
            message=(
                f"Backtest tamamlandı. "
                f"Toplam getiri: %{metrics['total_return_pct']:.2f} | "
                f"Sharpe: {metrics['sharpe_ratio']:.4f}"
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /feature-importance
# ---------------------------------------------------------------------------
@app.get("/feature-importance", tags=["Model"])
def feature_importance(
    asset: str = Query("BTC", description="Altın | USD/TRY | BTC"),
    top_n: int = Query(10, description="Gösterilecek feature sayısı"),
):
    """Eğitilmiş modelin feature importance tablosunu döner."""
    validate_asset(asset)

    if not model_exists(asset):
        raise HTTPException(
            status_code=404,
            detail=f"'{asset}' için eğitilmiş model bulunamadı. Önce /train endpoint'ini kullanın.",
        )

    try:
        model = load_model(asset)
        feature_cols = get_feature_columns()

        # En fazla mevcut feature sayısı kadar göster
        top_n = min(top_n, len(feature_cols))
        imp_df = get_feature_importance(model, feature_cols, top_n=top_n)

        return {
            "asset": asset,
            "top_n": top_n,
            "features": imp_df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /backtest-detail
# ---------------------------------------------------------------------------
@app.get("/backtest-detail", tags=["Backtest"])
def backtest_detail(
    asset: str = Query("BTC"),
    start: str = Query("2018-01-01"),
    end: Optional[str] = Query(None),
    initial_capital: Optional[float] = Query(None),
    limit: int = Query(100, description="Döndürülecek maksimum satır sayısı"),
):
    """
    Backtest'in gün gün detaylı sonuçlarını döner.
    Grafik çizmek için kullanılır.
    """
    validate_asset(asset)

    try:
        raw_df = get_data(asset, start=start, end=end)
        feat_df = build_features(raw_df)
        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in feat_df.columns]

        if model_exists(asset):
            model = load_model(asset)
        else:
            result = full_train_pipeline(
                df=feat_df,
                feature_cols=feature_cols,
                asset=asset,
                model_type="xgboost",
                optimize=False,
            )
            model = result["model"]

        _, X_test, _, _ = time_series_split(feat_df, feature_cols)
        split_idx = int(len(feat_df) * 0.80)
        df_test = feat_df.iloc[split_idx:].copy()

        preds = model.predict(X_test)
        cap = initial_capital or (INITIAL_CAPITAL_TRY if asset == "Altın" else INITIAL_CAPITAL_USD)

        bt_df = run_backtest(df_test, preds, asset=asset, initial_capital=cap)
        bh_df = buy_and_hold(df_test, asset=asset, initial_capital=cap)

        # Birleştir
        combined = bt_df.merge(bh_df, on="Date", how="left")

        # Limit uygula
        combined = combined.tail(limit)
        combined["Date"] = combined["Date"].astype(str)

        return {
            "asset": asset,
            "rows": len(combined),
            "data": combined.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Uygulama başlatma (direkt çalıştırma için)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
