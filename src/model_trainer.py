import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAMES = {
    "Altın":   "xau_usd_model.pkl",
    "USD/TRY": "usd_try_model.pkl",
    "BTC":     "btc_usd_model.pkl",
}

# ---------------------------------------------------------------------------
# Train / Test Split (zaman serisine göre, shuffle YOK)
# ---------------------------------------------------------------------------

def time_series_split(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Target",
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Zaman sırasına göre train/test bölümlemesi yapar.
    Shuffle kullanılmaz — gelecek verisinin modele sızmasını önler.

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(df) * train_ratio)

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model Tanımları
# ---------------------------------------------------------------------------

def get_random_forest(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgboost(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    optimize: bool = False,
) -> Any:
    """
    Modeli eğitir. İsteğe bağlı olarak GridSearch ile optimize eder.

    Args:
        X_train    : Eğitim feature'ları
        y_train    : Eğitim hedef değişkeni
        model_type : "random_forest" veya "xgboost"
        optimize   : True ise GridSearchCV uygular (yavaş)

    Returns:
        Eğitilmiş model
    """
    if model_type == "random_forest":
        model = get_random_forest()

        if optimize:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 8, 12],
                "min_samples_split": [5, 10, 20],
            }
            gs = GridSearchCV(
                model, param_grid,
                cv=3, scoring="accuracy",
                n_jobs=-1, verbose=0,
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            print(f"[RF] En iyi parametreler: {gs.best_params_}")
        else:
            model.fit(X_train, y_train)

    elif model_type == "xgboost":
        model = get_xgboost()

        if optimize:
            param_grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
            }
            gs = GridSearchCV(
                model, param_grid,
                cv=3, scoring="accuracy",
                n_jobs=-1, verbose=0,
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            print(f"[XGB] En iyi parametreler: {gs.best_params_}")
        else:
            model.fit(X_train, y_train)

    else:
        raise ValueError(f"Geçersiz model türü: '{model_type}'. 'random_forest' veya 'xgboost' kullanın.")

    return model


# ---------------------------------------------------------------------------
# Değerlendirme
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Test seti üzerinde modeli değerlendirir.

    Returns:
        {
          "accuracy"            : float,
          "directional_accuracy": float,
          "report"              : str,
          "confusion_matrix"    : np.ndarray,
          "y_pred"              : np.ndarray,
          "y_proba"             : np.ndarray,
        }
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 'Al' (1) olasılığı

    accuracy = accuracy_score(y_test, y_pred)

    # Directional Accuracy: tahmin yönü doğruysa 1
    directional_accuracy = float(np.mean(y_pred == y_test.values))

    report = classification_report(y_test, y_pred, target_names=["Sat (0)", "Al (1)"])
    cm     = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy":             round(accuracy, 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "report":               report,
        "confusion_matrix":     cm,
        "y_pred":               y_pred,
        "y_proba":              y_proba,
    }

    return results


def get_feature_importance(
    model: Any,
    feature_cols: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Feature importance tablosunu döner (RF veya XGBoost için).

    Returns:
        'feature' ve 'importance' kolonlarına sahip DataFrame (büyükten küçüğe sıralı)
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise AttributeError("Modelin 'feature_importances_' özelliği yok.")

    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return imp_df


# ---------------------------------------------------------------------------
# Kaydet / Yükle
# ---------------------------------------------------------------------------

def save_model(model: Any, asset: str) -> str:
    """
    Eğitilmiş modeli .pkl olarak kaydeder.

    Returns:
        Kaydedilen dosyanın tam yolu
    """
    if asset not in MODEL_FILENAMES:
        raise ValueError(f"Geçersiz varlık: '{asset}'")

    filepath = os.path.join(MODELS_DIR, MODEL_FILENAMES[asset])
    joblib.dump(model, filepath)
    print(f"[✓] Model kaydedildi: {filepath}")
    return filepath


def load_model(asset: str) -> Any:
    """
    Kaydedilmiş modeli yükler.

    Returns:
        Yüklenmiş model nesnesi
    """
    if asset not in MODEL_FILENAMES:
        raise ValueError(f"Geçersiz varlık: '{asset}'")

    filepath = os.path.join(MODELS_DIR, MODEL_FILENAMES[asset])

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{asset}' için kayıtlı model bulunamadı: {filepath}\n"
            f"Lütfen önce modeli eğitin."
        )

    model = joblib.load(filepath)
    print(f"[✓] Model yüklendi: {filepath}")
    return model


def model_exists(asset: str) -> bool:
    """Modelin daha önce eğitilip kaydedilip kaydedilmediğini kontrol eder."""
    if asset not in MODEL_FILENAMES:
        return False
    filepath = os.path.join(MODELS_DIR, MODEL_FILENAMES[asset])
    return os.path.exists(filepath)


# ---------------------------------------------------------------------------
# Tam Pipeline (veri → eğitim → değerlendirme → kaydet)
# ---------------------------------------------------------------------------

def full_train_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    asset: str,
    model_type: str = "xgboost",
    train_ratio: float = 0.80,
    optimize: bool = False,
    target_col: str = "Target",
) -> Dict[str, Any]:
    """
    Uçtan uca eğitim pipeline'ı:
      1. Train/test split
      2. Model eğitimi
      3. Değerlendirme
      4. Kaydetme

    Returns:
        {
          "model"    : eğitilmiş model,
          "metrics"  : değerlendirme sonuçları,
          "X_test"   : pd.DataFrame,
          "y_test"   : pd.Series,
          "df_train" : eğitim DataFrame'i,
          "df_test"  : test DataFrame'i,
        }
    """
    print(f"\n{'='*55}")
    print(f"  {asset} | {model_type.upper()} | optimize={optimize}")
    print(f"{'='*55}")

    # 1. Split
    X_train, X_test, y_train, y_test = time_series_split(
        df, feature_cols, target_col, train_ratio
    )
    print(f"  Train: {len(X_train)} satır | Test: {len(X_test)} satır")

    # 2. Eğitim
    model = train_model(X_train, y_train, model_type=model_type, optimize=optimize)

    # 3. Değerlendirme
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Directional Acc.  : {metrics['directional_accuracy']:.4f}")
    print(f"\n{metrics['report']}")

    # 4. Kaydet
    save_model(model, asset)

    # Eğitim/test DataFrame indekslerini al
    split_idx = int(len(df) * train_ratio)
    df_train  = df.iloc[:split_idx].copy()
    df_test   = df.iloc[split_idx:].copy()

    return {
        "model":    model,
        "metrics":  metrics,
        "X_test":   X_test,
        "y_test":   y_test,
        "df_train": df_train,
        "df_test":  df_test,
    }


# ---------------------------------------------------------------------------
# Hızlı test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_fetcher import get_data
    from src.feature_engineering import build_features, get_feature_columns

    asset = "BTC"
    print(f"Test: {asset} verisi çekiliyor...")

    raw = get_data(asset, start="2018-01-01")
    df  = build_features(raw)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]

    result = full_train_pipeline(
        df=df,
        feature_cols=feature_cols,
        asset=asset,
        model_type="xgboost",
        optimize=False,
    )

    imp = get_feature_importance(result["model"], feature_cols)
    print("\nTop Feature Importances:")
    print(imp)
