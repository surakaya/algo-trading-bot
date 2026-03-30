# 🤖 Algo Trading Bot – Altın / Döviz / Kripto

ML tabanlı yön tahmin modeli, backtesting simülasyonu ve Streamlit dashboard'u içeren FinTech projesi.

---

## 📌 Proje Özeti

| Özellik | Detay |
|---|---|
| **Varlıklar** | Altın (XAU/USD → TRY/gram), USD/TRY, BTC/USD |
| **Model** | XGBoost / Random Forest |
| **Feature'lar** | MA(5,20,50), RSI(14), Momentum, Lag, Bollinger Bands, Volatilite |
| **Backtest** | Al/Sat simülasyonu, Sharpe Ratio, Max Drawdown, Win Rate |
| **Dashboard** | Streamlit (Streamlit Cloud'a deploy edilebilir) |
| **API** | FastAPI (lokal çalışır, portfolyo için) |

---

## 📁 Klasör Yapısı

```
algo-trading-bot/
│
├── data/                        # Ham ve işlenmiş veriler
│   └── cache/                   # yfinance cache
│
├── models/                      # Eğitilmiş .pkl modeller
│   ├── xau_usd_model.pkl
│   ├── usd_try_model.pkl
│   └── btc_usd_model.pkl
│
├── src/                         # İş mantığı modülleri
│   ├── data_fetcher.py          # yfinance veri çekme
│   ├── feature_engineering.py  # Teknik indikatörler
│   ├── model_trainer.py         # Eğitim, değerlendirme, kaydetme
│   ├── backtester.py            # Backtest simülasyonu
│   └── predictor.py             # Birleşik tahmin arayüzü
│
├── api/
│   └── main.py                  # FastAPI backend
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Kurulum

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Streamlit Dashboard'u Başlat

```bash
streamlit run dashboard/app.py
```

### 3. FastAPI Backend'i Başlat (opsiyonel, lokal)

```bash
uvicorn api.main:app --reload --port 8000
```

API Swagger dokümantasyonu: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖥️ Dashboard Kullanımı

1. **Sidebar'dan varlık seç** → Altın / USD/TRY / BTC
2. **Tarih aralığı belirle** → Başlangıç ve bitiş tarihi
3. **"Tahmin Et" butonuna tıkla**
4. Dashboard şunları gösterir:
   - Güncel fiyat grafiği (Close + MA çizgileri)
   - **AL / SAT / BEKLE** sinyali (renk + güven skoru)
   - RSI ve Momentum grafikleri
   - Backtest simülasyonu (model vs. Buy & Hold)
   - Performans metrikleri
   - Son 5 günün tahmin tablosu

---

## 🔌 API Endpoints

| Method | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Sağlık kontrolü |
| `GET` | `/predict?asset=BTC` | Yön tahmini |
| `POST` | `/train` | Model eğit |
| `GET` | `/backtest?asset=BTC` | Backtest özeti |
| `GET` | `/backtest-detail?asset=BTC` | Gün gün backtest |
| `GET` | `/feature-importance?asset=BTC` | Feature önemi |
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

## 🧠 ML Pipeline

```
Ham Veri (yfinance)
        ↓
Feature Engineering
  MA(5,20,50) | RSI(14) | Momentum | Lag(1,3)
  Bollinger Bands | Volatilite
        ↓
Zaman Serisi Split (%80 train / %20 test)
        ↓
XGBoost / Random Forest Eğitimi
        ↓
Değerlendirme (Accuracy, Directional Accuracy)
        ↓
Model Kaydet (.pkl)
        ↓
Tahmin → AL / SAT / BEKLE Sinyali
```

---

## 📊 Backtest Stratejisi

- **Başlangıç Sermayesi:** 10.000 ₺ (Altın) / $10.000 (USD/TRY, BTC)
- **İşlem Maliyeti:** %0.1 (her alım/satımda)
- **Strateji:** Tahmin = 1 (Al) → Pozisyon aç | Tahmin = 0 (Sat) → Pozisyon kapat
- **Karşılaştırma:** Buy & Hold benchmark ile karşılaştırılır

---

## ☁️ Streamlit Cloud Deploy

1. GitHub'a push et
2. [share.streamlit.io](https://share.streamlit.io) adresine git
3. Repository'i bağla
4. **Main file path:** `dashboard/app.py`
5. Deploy!

> **Not:** Deploy öncesi modelleri eğitip `models/` klasörünü repoya dahil et.

---

## 🛠️ Gereksinimler

- Python 3.10+
- İnternet bağlantısı (yfinance veri çekimi için)

---

## 📈 Desteklenen Varlıklar

| Varlık | Ticker | Para Birimi |
|---|---|---|
| Altın | GC=F (XAU/USD) | USD → TRY/gram |
| Döviz | TRY=X (USD/TRY) | TRY |
| Bitcoin | BTC-USD | USD |

---

## ⚠️ Sorumluluk Reddi

Bu proje **yalnızca eğitim amaçlıdır**. Gerçek finansal yatırım kararları için kullanılmamalıdır. Geçmiş performans gelecekteki sonuçları garanti etmez.

---

## 👤 Yazar

FinTech ML projesi — Algo Trading Bot  
*Streamlit Dashboard + FastAPI + XGBoost/Random Forest + Backtesting*