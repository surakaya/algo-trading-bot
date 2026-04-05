#  Algo Trading Bot

> ML tabanlı yön tahmin modeli, çoklu strateji backtesting simülasyonu ve canlı Streamlit dashboard'u içeren FinTech projesi.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algo-trading-bot.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Project Overview

Bu proje; **Altın (XAU/USD → TRY/gram)**, **USD/TRY** ve **BTC/USD** varlıkları üzerinde:

- Makine öğrenmesi ile fiyat yön tahmini yapar
- 3 farklı strateji ile gerçekçi backtest simülasyonu çalıştırır
- Sonuçları interaktif bir Streamlit dashboard üzerinden sunar
- FastAPI ile REST API endpoint'leri sağlar

---

##  Features

| Özellik | Detay |
|---|---|
| **Veri Kaynağı** | Yahoo Finance (`yfinance`) — 2015'ten bugüne |
| **Varlıklar** | Altın (XAU/USD → TRY/gram), USD/TRY, BTC/USD |
| **ML Modeller** | XGBoost, Random Forest — otomatik en iyisi seçilir |
| **Stratejiler** | ML Modeli, SMA Crossover, RSI Stratejisi |
| **Backtest** | Transaction cost + Slippage dahil gerçekçi simülasyon |
| **Metrikler** | Sharpe Ratio, Max Drawdown, Win Rate, Annualized Return |
| **Dashboard** | Streamlit — interaktif grafikler, tahmin sinyali |
| **API** | FastAPI — `/predict`, `/train`, `/backtest` endpoint'leri |

---

##  Strategies

###  ML Modeli
XGBoost ve Random Forest modellerini eğitir, test accuracy'si yüksek olanı otomatik seçer. Teknik indikatörlerden (MA, RSI, Momentum, Bollinger Bands) öğrenir ve yarınki fiyat yönünü tahmin eder.

```
Feature Engineering → Train/Test Split → XGBoost vs RF → Kazananı Seç → Tahmin
```

###  SMA Crossover
Kısa ve uzun vadeli hareketli ortalamaların kesişimine dayalı klasik trend takip stratejisi.

```
MA5 > MA20 → AL
MA5 ≤ MA20 → SAT
```

###  RSI Stratejisi
RSI göstergesinin aşırı alım/satım bölgelerine dayalı kontrarian strateji.

```
RSI < 30 → AL  (aşırı satım, geri dönüş beklenir)
RSI > 70 → SAT (aşırı alım, düzeltme beklenir)
Arada    → Önceki sinyali koru
```

---

##  Performance Metrics

| Metrik | Açıklama |
|---|---|
| **Total Return** | Başlangıç sermayesine göre toplam getiri (%) |
| **Annualized Return** | Yıllık bileşik getiri (%) |
| **Sharpe Ratio** | Risk-adjusted getiri — yüksek = iyi (>1 hedeflenir) |
| **Max Drawdown** | Zirve noktasından en büyük gerileme (%) |
| **Win Rate** | Kârlı işlemlerin toplam işlemlere oranı (%) |
| **Buy & Hold vs Strategy** | Pasif tutma stratejisi ile karşılaştırma |

---

##  Feature Engineering

Model eğitiminde kullanılan teknik indikatörler:

| Feature | Açıklama |
|---|---|
| `MA_5, MA_20, MA_50` | Hareketli ortalamalar (5, 20, 50 gün) |
| `RSI_14` | Relative Strength Index (14 gün) |
| `Momentum` | `Close(t) / Close(t-5)` — fiyat ivmesi |
| `Lag_1, Lag_3` | 1 ve 3 günlük gecikmeli kapanış fiyatları |
| `BB_Upper/Lower/Mid` | Bollinger Bands (20 gün, 2σ) |
| `BB_Pct` | Fiyatın bantlar içindeki yüzde konumu |
| `Volatility` | 20 günlük yıllıklaştırılmış volatilite |

---

##  Backtest Gerçekçiliği

Sonuçların gerçek dünya koşullarını yansıtması için:

```python
TRANSACTION_COST = 0.001   # %0.1 — her alım/satımda
SLIPPAGE         = 0.0005  # %0.05 — alımda pahalıya, satışta ucuza
```

- **Alımda:** `buy_price  = market_price × (1 + slippage)`
- **Satışta:** `sell_price = market_price × (1 − slippage)`
- **Her işlemde:** ayrıca transaction cost düşülür

---

##  Demo

🔗 **[algo-trading-bot.streamlit.app](https://algo-trading-bot.streamlit.app)**

**Kullanım:**
1. Sidebar'dan varlık seç: BTC / Altın / USD-TRY
2. Strateji seç: ML Modeli / SMA Crossover / RSI Stratejisi
3. Tarih aralığı belirle
4. **Tahmin Et & Analiz** butonuna bas


##  API Endpoints

| Method | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Sağlık kontrolü |
| `GET` | `/predict?asset=BTC` | Yön tahmini |
| `POST` | `/train` | Model eğit & kaydet |
| `GET` | `/backtest?asset=BTC` | Backtest özeti |
| `GET` | `/backtest-detail?asset=BTC` | Gün gün backtest verisi |
| `GET` | `/feature-importance?asset=BTC` | Feature önem sıralaması |
| `GET` | `/latest-price?asset=BTC` | Güncel fiyat |

### Örnek `/predict` Yanıtı
```json
{
  "ticker": "BTC-USD",
  "asset": "BTC",
  "prediction": "up",
  "signal": "Al",
  "confidence": 0.72,
  "current_price": 67432.5,
  "try_gram_price": null,
  "date": "2024-12-01"
}
```

---

##  Proje Yapısı

```
algo-trading-bot/
│
├── data/cache/              # yfinance önbellek
├── models/                  # Eğitilmiş .pkl modeller
│
├── src/
│   ├── data_fetcher.py      # Veri çekme (yfinance)
│   ├── feature_engineering.py  # Teknik indikatörler
│   ├── model_trainer.py     # Model eğitimi & değerlendirme
│   ├── strategies.py        # SMA / RSI / ML sinyal üretici
│   ├── backtester.py        # Backtest simülasyonu
│   └── predictor.py         # Birleşik tahmin pipeline'ı
│
├── api/main.py              # FastAPI backend
├── dashboard/app.py         # Streamlit dashboard
│
├── requirements.txt
├── runtime.txt              # Python 3.11
└── README.md
```

---

##  Teknoloji Stack'i

| Katman | Teknoloji |
|---|---|
| **Veri** | yfinance, pandas, numpy |
| **ML** | scikit-learn, XGBoost |
| **Backtest** | Özel simülasyon motoru |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, uvicorn |
| **Deploy** | Streamlit Community Cloud |

---

## ⚠️ Sorumluluk Reddi

Bu proje **yalnızca eğitim ve portfolyo amaçlıdır**. Gerçek finansal yatırım kararlarında kullanılmamalıdır. Geçmiş performans gelecekteki sonuçları garanti etmez.

---

*FinTech ML projesi — Streamlit Dashboard + FastAPI + XGBoost/Random Forest + Multi-Strategy Backtesting*
