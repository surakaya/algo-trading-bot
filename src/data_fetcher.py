import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# Rate limit retry ayarları
MAX_RETRIES = 3
RETRY_WAIT  = 10  # saniye (her denemede 2 katına çıkar)

# Ticker sabitleri
TICKERS = {
    "Altın":   "GC=F",
    "USD/TRY": "TRY=X",
    "BTC":     "BTC-USD",
}

# Altın TRY/gram dönüşümü için kur
USDTRY_TICKER = "TRY=X"
TROY_OUNCE_TO_GRAM = 31.1035


def fetch_raw_data(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    yfinance üzerinden ham OHLCV verisini çeker.

    Args:
        ticker : yfinance ticker sembolü (ör: 'GC=F')
        start  : Başlangıç tarihi (YYYY-MM-DD)
        end    : Bitiş tarihi (YYYY-MM-DD), None ise bugün

    Returns:
        OHLCV kolonlarına sahip pandas DataFrame
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )

            # yfinance 0.2+ multi-level kolonları düzleştir
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                raise ValueError(f"'{ticker}' için veri çekilemedi. Ticker doğru mu?")

            break  # başarılı → döngüden çık

        except Exception as e:
            last_exc = e
            err_str  = str(e).lower()
            is_rate_limit = any(
                x in err_str for x in ["rate", "too many", "429", "limit", "ratelimit"]
            )
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT * (2 ** attempt)  # 10s, 20s, 40s
                print(f"[yfinance] Rate limit — {wait}s bekleniyor... (deneme {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

    if last_exc and df.empty:
        raise last_exc

    # Sadece ihtiyacımız olan kolonları al
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # NaN içeren satırları düşür
    df.dropna(subset=["Close"], inplace=True)

    return df


def fetch_usdtry(
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """
    USD/TRY günlük kapanış fiyatlarını çeker.

    Returns:
        'USDTRY' adlı pandas Series
    """
    df = fetch_raw_data(USDTRY_TICKER, start=start, end=end)
    series = df["Close"].rename("USDTRY")
    return series


def fetch_gold_try_gram(
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    XAU/USD verisini çekip TRY/gram'a dönüştürür.
    Dönüşüm: (XAU_USD_Close * USDTRY) / 31.1035

    Returns:
        OHLCV + Close_TRY_gram kolonlarına sahip DataFrame
    """
    gold_df = fetch_raw_data("GC=F", start=start, end=end)
    usdtry  = fetch_usdtry(start=start, end=end)

    # Ortak tarihleri hizala
    combined = gold_df.join(usdtry, how="left")
    combined["USDTRY"].ffill(inplace=True)  # eksik kurları forward-fill ile doldur

    combined["Close_TRY_gram"] = (
        combined["Close"] * combined["USDTRY"]
    ) / TROY_OUNCE_TO_GRAM

    # Model için kullanılacak ana fiyat XAU/USD olacak (Close kolonu)
    # TRY/gram sadece display için
    combined.drop(columns=["USDTRY"], inplace=True)
    combined.dropna(subset=["Close", "Close_TRY_gram"], inplace=True)

    return combined


def get_data(
    asset: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ana veri çekme fonksiyonu. Dashboard ve API buraya bağlanır.

    Args:
        asset : "Altın", "USD/TRY" veya "BTC"
        start : Başlangıç tarihi
        end   : Bitiş tarihi (None ise bugün)

    Returns:
        Hazır pandas DataFrame
    """
    if asset not in TICKERS:
        raise ValueError(
            f"Geçersiz varlık: '{asset}'. Seçenekler: {list(TICKERS.keys())}"
        )

    if asset == "Altın":
        df = fetch_gold_try_gram(start=start, end=end)
    else:
        ticker = TICKERS[asset]
        df = fetch_raw_data(ticker, start=start, end=end)

    return df


def get_latest_price(asset: str) -> dict:
    """
    Seçilen varlığın en güncel fiyat bilgisini döner.

    Returns:
        {
          "asset"     : str,
          "price"     : float,   # USD cinsinden (XAU/USD veya BTC/USD veya USDTRY)
          "try_gram"  : float | None,   # Sadece altın için
          "date"      : str,
        }
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")

    df = get_data(asset, start=start, end=end)
    latest = df.iloc[-1]

    result = {
        "asset": asset,
        "price": round(float(latest["Close"]), 4),
        "try_gram": round(float(latest["Close_TRY_gram"]), 2)
        if "Close_TRY_gram" in df.columns
        else None,
        "date": str(df.index[-1].date()),
    }
    return result


# ---------------------------------------------------------------------------
# Hızlı test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for asset in ["Altın", "USD/TRY", "BTC"]:
        print(f"\n{'='*50}")
        print(f"Varlık: {asset}")
        df = get_data(asset, start="2023-01-01")
        print(df.tail(3))
        print(f"Shape: {df.shape}")
        info = get_latest_price(asset)
        print(f"Güncel: {info}")
