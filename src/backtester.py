import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
INITIAL_CAPITAL_TRY = 10_000.0   # Altın için başlangıç sermayesi (TRY)
INITIAL_CAPITAL_USD = 10_000.0   # USD/TRY ve BTC için başlangıç sermayesi (USD)
TRANSACTION_COST    = 0.001       # İşlem maliyeti: %0.1 (her alım/satımda)
SLIPPAGE            = 0.0005      # Kayma payı: %0.05 (piyasa fiyatı sapması)


# ---------------------------------------------------------------------------
# Ana Backtest Fonksiyonu
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    predictions: np.ndarray,
    asset: str = "BTC",
    initial_capital: Optional[float] = None,
    slippage: float = SLIPPAGE,
) -> pd.DataFrame:
    """
    Modelin ürettiği tahminlere göre gerçekçi bir Al/Sat simülasyonu çalıştırır.

    Strateji:
        - Tahmin = 1 (fiyat çıkacak) → pozisyon aç (al)
        - Tahmin = 0 (fiyat düşecek) → pozisyon kapat (sat) ya da bekle

    Gerçekçilik:
        - Her işlemde transaction cost (%0.1) uygulanır
        - Her işlemde slippage (kayma payı, %0.05) uygulanır:
            Alımda: fiyat * (1 + slippage)  → biraz pahalıya alırsın
            Satışta: fiyat * (1 - slippage) → biraz ucuza satarsın

    Args:
        df             : build_features() çıktısı (Date ve Close kolonları olmalı)
        predictions    : Model tahminleri (0/1 array, df ile aynı uzunlukta)
        asset          : "Altın", "USD/TRY" veya "BTC"
        initial_capital: Başlangıç sermayesi (None ise varlığa göre otomatik)
        slippage       : Kayma payı oranı (varsayılan: 0.0005 = %0.05)

    Returns:
        Her satırda bir işlem günü olan backtest sonuç DataFrame'i.
        Kolonlar:
            Date, Close, Signal, Position, Portfolio_Value,
            Daily_Return, Cumulative_Return, Trade_PnL
    """
    if initial_capital is None:
        initial_capital = (
            INITIAL_CAPITAL_TRY if asset == "Altın" else INITIAL_CAPITAL_USD
        )

    # Fiyat kolonunu belirle
    price_col = "Close_TRY_gram" if (asset == "Altın" and "Close_TRY_gram" in df.columns) else "Close"

    # Sadece ihtiyacımız olan kolonları al
    result = df[["Date", price_col]].copy().rename(columns={price_col: "Close"})
    result = result.reset_index(drop=True)

    n = len(result)
    if len(predictions) != n:
        raise ValueError(
            f"Tahmin sayısı ({len(predictions)}) veri satır sayısıyla ({n}) eşleşmiyor."
        )

    result["Signal"]    = predictions.astype(int)   # 1=Al, 0=Sat/Bekle
    result["Position"]  = 0.0    # Anlık pozisyon (1=pozisyonda, 0=nakit)
    result["Portfolio_Value"]   = initial_capital
    result["Daily_Return"]      = 0.0
    result["Cumulative_Return"] = 0.0
    result["Trade_PnL"]         = 0.0
    result["Trade"]             = ""    # "BUY" / "SELL" / ""

    cash          = initial_capital
    units_held    = 0.0     # Elde tutulan varlık miktarı
    in_position   = False

    for i in range(n):
        price  = float(result.at[i, "Close"])
        signal = int(result.at[i, "Signal"])

        trade_pnl = 0.0
        trade_tag = ""

        # --- AL sinyali ---
        if signal == 1 and not in_position and cash > 0:
            buy_price    = price * (1 + slippage)   # slippage: biraz pahalıya al
            cost         = cash * TRANSACTION_COST
            invest       = cash - cost
            units_held   = invest / buy_price
            cash         = 0.0
            in_position  = True
            trade_tag    = "BUY"

        # --- SAT sinyali ---
        elif signal == 0 and in_position:
            sell_price   = price * (1 - slippage)   # slippage: biraz ucuza sat
            gross        = units_held * sell_price
            fee          = gross * TRANSACTION_COST
            net          = gross - fee
            trade_pnl    = net - (
                result.at[i - 1, "Portfolio_Value"] if i > 0 else initial_capital
            )
            cash         = net
            units_held   = 0.0
            in_position  = False
            trade_tag    = "SELL"

        # Portföy değeri: nakit + pozisyondaki varlığın güncel değeri
        portfolio_value = cash + (units_held * price if in_position else 0.0)

        result.at[i, "Position"]        = 1.0 if in_position else 0.0
        result.at[i, "Portfolio_Value"] = round(portfolio_value, 4)
        result.at[i, "Trade_PnL"]       = round(trade_pnl, 4)
        result.at[i, "Trade"]           = trade_tag

    # Günlük getiri
    result["Daily_Return"] = result["Portfolio_Value"].pct_change().fillna(0.0)

    # Kümülatif getiri (%)
    result["Cumulative_Return"] = (
        (result["Portfolio_Value"] / initial_capital - 1) * 100
    ).round(4)

    return result


# ---------------------------------------------------------------------------
# Performans Metrikleri
# ---------------------------------------------------------------------------
def compute_metrics(backtest_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Backtest sonucu üzerinden temel performans metriklerini hesaplar.

    Args:
        backtest_df    : run_backtest() çıktısı
        initial_capital: Başlangıç sermayesi

    Returns:
        Metrik sözlüğü
    """
    final_value = backtest_df["Portfolio_Value"].iloc[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100

    # Buy & Hold karşılaştırması
    buy_hold_return_pct = (
        (backtest_df["Close"].iloc[-1] / backtest_df["Close"].iloc[0]) - 1
    ) * 100

    # Yıllıklaştırılmış getiri
    n_days = len(backtest_df)
    n_years = n_days / 252
    annualized_return = (
        ((final_value / initial_capital) ** (1 / n_years) - 1) * 100
        if n_years > 0
        else 0.0
    )

    # Sharpe Ratio (risk-free rate = 0 varsayımıyla)
    daily_returns = backtest_df["Daily_Return"]
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    # Max Drawdown
    rolling_max  = backtest_df["Portfolio_Value"].cummax()
    drawdown     = (backtest_df["Portfolio_Value"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # İşlem sayıları
    trades      = backtest_df[backtest_df["Trade"] != ""]
    num_buys    = (trades["Trade"] == "BUY").sum()
    num_sells   = (trades["Trade"] == "SELL").sum()
    winning_trades = (backtest_df["Trade_PnL"] > 0).sum()
    losing_trades  = (backtest_df["Trade_PnL"] < 0).sum()

    win_rate = (
        (winning_trades / num_sells * 100) if num_sells > 0 else 0.0
    )

    return {
        "initial_capital"      : round(initial_capital, 2),
        "final_value"          : round(final_value, 2),
        "total_return_pct"     : round(total_return_pct, 2),
        "buy_hold_return_pct"  : round(buy_hold_return_pct, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "sharpe_ratio"         : round(sharpe, 4),
        "max_drawdown_pct"     : round(max_drawdown, 2),
        "num_buys"             : int(num_buys),
        "num_sells"            : int(num_sells),
        "winning_trades"       : int(winning_trades),
        "losing_trades"        : int(losing_trades),
        "win_rate_pct"         : round(win_rate, 2),
    }


# ---------------------------------------------------------------------------
# Yardımcı: Basit Buy & Hold Karşılaştırması
# ---------------------------------------------------------------------------
def buy_and_hold(
    df: pd.DataFrame,
    asset: str = "BTC",
    initial_capital: Optional[float] = None,
) -> pd.DataFrame:
    """
    Karşılaştırma için basit buy & hold stratejisi simüle eder.
    Tüm sermaye ilk günde alınır, son güne kadar tutulur.

    Returns:
        Date ve BH_Portfolio_Value kolonlarına sahip DataFrame
    """
    if initial_capital is None:
        initial_capital = (
            INITIAL_CAPITAL_TRY if asset == "Altın" else INITIAL_CAPITAL_USD
        )

    price_col = "Close_TRY_gram" if (asset == "Altın" and "Close_TRY_gram" in df.columns) else "Close"

    result = df[["Date", price_col]].copy().rename(columns={price_col: "Close"})
    result = result.reset_index(drop=True)

    first_price = float(result["Close"].iloc[0])
    fee         = initial_capital * TRANSACTION_COST
    units       = (initial_capital - fee) / first_price

    result["BH_Portfolio_Value"] = (units * result["Close"]).round(4)
    result["BH_Cumulative_Return"] = (
        (result["BH_Portfolio_Value"] / initial_capital - 1) * 100
    ).round(4)

    return result[["Date", "BH_Portfolio_Value", "BH_Cumulative_Return"]]


# ---------------------------------------------------------------------------
# Hızlı Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_fetcher import get_data
    from src.feature_engineering import build_features

    print("Backtest testi başlıyor...")
    df_raw    = get_data("BTC", start="2022-01-01")
    df_feat   = build_features(df_raw)

    # Rastgele tahminlerle test
    np.random.seed(42)
    preds = np.random.randint(0, 2, size=len(df_feat))

    result = run_backtest(df_feat, preds, asset="BTC")
    print(result.tail(5))

    metrics = compute_metrics(result, initial_capital=INITIAL_CAPITAL_USD)
    print("\nPerformans Metrikleri:")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    bh = buy_and_hold(df_feat, asset="BTC")
    print("\nBuy & Hold (son 5 satır):")
    print(bh.tail(5))
