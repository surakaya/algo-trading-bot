import sys
import os
import warnings
import time
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date

from src.data_fetcher import get_data, get_latest_price
from src.feature_engineering import build_features, get_feature_columns
from src.predictor import predict_next_day, predict_multi_day, run_full_backtest

# ---------------------------------------------------------------------------
# Sayfa Ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Algo Trading Bot – Altın / Döviz / Kripto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Genel arka plan */
        .main { background-color: #FFFFFF; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }

        /* Başlık */
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1a1a2e;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            font-size: 1rem;
            color: #555;
            margin-top: -8px;
            margin-bottom: 1.2rem;
        }

        /* Metrik kartlar */
        .metric-card {
            background: #f8f9ff;
            border: 1.5px solid #e0e4f0;
            border-radius: 12px;
            padding: 18px 22px;
            text-align: center;
        }
        .metric-card .label {
            font-size: 0.78rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        .metric-card .value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1a1a2e;
            margin: 4px 0 0 0;
        }
        .metric-card .delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 2px;
        }

        /* Sinyal kutuları */
        .signal-al {
            background: #E3F2FD;
            border: 2.5px solid #1976D2;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }
        .signal-sat {
            background: #FFEBEE;
            border: 2.5px solid #D32F2F;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }
        .signal-bekle {
            background: #FFF8E1;
            border: 2.5px solid #F57C00;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }
        .signal-text {
            font-size: 2.8rem;
            font-weight: 900;
            letter-spacing: 2px;
        }
        .signal-sub {
            font-size: 0.9rem;
            color: #555;
            margin-top: 6px;
        }

        /* Info banner */
        .info-banner {
            background: #E3F2FD;
            border-left: 4px solid #1976D2;
            border-radius: 6px;
            padding: 10px 16px;
            color: #1565C0;
            font-size: 0.9rem;
            margin-bottom: 12px;
        }
        .warn-banner {
            background: #FFEBEE;
            border-left: 4px solid #D32F2F;
            border-radius: 6px;
            padding: 10px 16px;
            color: #B71C1C;
            font-size: 0.9rem;
            margin-bottom: 12px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f4f6fb;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stDateInput label,
        section[data-testid="stSidebar"] .stRadio label {
            font-weight: 600;
            color: #1a1a2e;
        }

        /* Divider */
        hr { border-color: #e0e4f0; }

        /* Tablo */
        .signal-table td, .signal-table th {
            padding: 8px 14px;
            border-bottom: 1px solid #e0e4f0;
            font-size: 0.9rem;
        }
        .signal-table th {
            background: #f4f6fb;
            font-weight: 700;
            color: #1a1a2e;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #aaa;
            font-size: 0.78rem;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #eee;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def fmt_price(value: float, asset: str) -> str:
    if asset == "USD/TRY":
        return f"₺{value:,.4f}"
    elif asset == "Altın":
        return f"${value:,.2f}"
    else:
        return f"${value:,.2f}"


def fmt_try_gram(value: float) -> str:
    return f"₺{value:,.2f} / gram"


def color_return(val: float) -> str:
    return "#D32F2F" if val < 0 else "#2E7D32"


def make_price_chart(feature_df: pd.DataFrame, asset: str) -> go.Figure:
    """Fiyat + MA grafigi."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=feature_df["Date"], y=feature_df["Close"],
        name="Kapanış", line=dict(color="#1976D2", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Fiyat: %{y:,.4f}<extra></extra>",
    ))
    if "MA_5" in feature_df.columns:
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["MA_5"],
            name="MA 5", line=dict(color="#43A047", width=1.5, dash="dot"),
            hovertemplate="MA5: %{y:,.4f}<extra></extra>",
        ))
    if "MA_20" in feature_df.columns:
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["MA_20"],
            name="MA 20", line=dict(color="#FB8C00", width=1.5, dash="dash"),
            hovertemplate="MA20: %{y:,.4f}<extra></extra>",
        ))
    if "MA_50" in feature_df.columns:
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["MA_50"],
            name="MA 50", line=dict(color="#8E24AA", width=1.5, dash="longdash"),
            hovertemplate="MA50: %{y:,.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"{asset} – Fiyat & Hareketli Ortalamalar", font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tarih"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Fiyat"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_rsi_chart(feature_df: pd.DataFrame) -> go.Figure:
    """RSI grafigi."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feature_df["Date"], y=feature_df["RSI_14"],
        name="RSI 14", line=dict(color="#1976D2", width=2),
        fill="tozeroy", fillcolor="rgba(25,118,210,0.08)",
    ))
    # Overbought / Oversold çizgileri
    fig.add_hline(y=70, line=dict(color="#D32F2F", dash="dash", width=1.5),
                  annotation_text="Aşırı Alım (70)", annotation_position="top right",
                  annotation_font_color="#D32F2F")
    fig.add_hline(y=30, line=dict(color="#2E7D32", dash="dash", width=1.5),
                  annotation_text="Aşırı Satım (30)", annotation_position="bottom right",
                  annotation_font_color="#2E7D32")
    fig.add_hline(y=50, line=dict(color="#aaa", dash="dot", width=1), )

    fig.update_layout(
        title=dict(text="RSI (14)", font=dict(size=15, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[0, 100]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=280,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
    )
    return fig


def make_momentum_chart(feature_df: pd.DataFrame) -> go.Figure:
    """Momentum grafigi."""
    colors = ["#2E7D32" if v >= 1 else "#D32F2F" for v in feature_df["Momentum"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_df["Date"], y=feature_df["Momentum"] - 1,
        marker_color=colors,
        name="Momentum",
        hovertemplate="%{x|%Y-%m-%d}<br>Momentum: %{customdata:.4f}<extra></extra>",
        customdata=feature_df["Momentum"],
    ))
    fig.add_hline(y=0, line=dict(color="#1a1a2e", width=1))
    fig.update_layout(
        title=dict(text="Momentum (Close / Close[t-5] - 1)", font=dict(size=15, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=280,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
    )
    return fig


def make_backtest_chart(backtest_df: pd.DataFrame, bh_df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Backtest portföy değeri grafigi."""
    fig = go.Figure()

    # Strateji çizgisi
    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Portfolio_Value"],
        name="Strateji", line=dict(color="#1976D2", width=2.5),
        fill="tozeroy", fillcolor="rgba(25,118,210,0.06)",
        hovertemplate="%{x|%Y-%m-%d}<br>Portföy: %{y:,.2f}<extra></extra>",
    ))

    # Buy & Hold çizgisi
    if "BH_Portfolio_Value" in bh_df.columns:
        fig.add_trace(go.Scatter(
            x=bh_df["Date"], y=bh_df["BH_Portfolio_Value"],
            name="Buy & Hold", line=dict(color="#888", width=1.8, dash="dash"),
            hovertemplate="B&H: %{y:,.2f}<extra></extra>",
        ))

    # Başlangıç sermayesi çizgisi
    fig.add_hline(
        y=initial_capital,
        line=dict(color="#D32F2F", dash="dot", width=1.5),
        annotation_text=f"Başlangıç: {initial_capital:,.0f}",
        annotation_position="top left",
        annotation_font_color="#D32F2F",
    )

    # Al/Sat işaretleri
    buys  = backtest_df[backtest_df["Trade"] == "BUY"]
    sells = backtest_df[backtest_df["Trade"] == "SELL"]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Date"], y=buys["Portfolio_Value"],
            mode="markers", name="AL",
            marker=dict(symbol="triangle-up", size=10, color="#1976D2"),
            hovertemplate="AL: %{x|%Y-%m-%d}<extra></extra>",
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Date"], y=sells["Portfolio_Value"],
            mode="markers", name="SAT",
            marker=dict(symbol="triangle-down", size=10, color="#D32F2F"),
            hovertemplate="SAT: %{x|%Y-%m-%d}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Backtest – Portföy Değeri", font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tarih"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Portföy Değeri"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_cumulative_return_chart(backtest_df: pd.DataFrame, bh_df: pd.DataFrame) -> go.Figure:
    """Kümülatif getiri (%) grafigi."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Cumulative_Return"],
        name="Strateji (%)",
        line=dict(color="#1976D2", width=2),
        fill="tozeroy",
        fillcolor="rgba(25,118,210,0.07)",
        hovertemplate="%{x|%Y-%m-%d}<br>Getiri: %{y:.2f}%<extra></extra>",
    ))
    if "BH_Cumulative_Return" in bh_df.columns:
        fig.add_trace(go.Scatter(
            x=bh_df["Date"], y=bh_df["BH_Cumulative_Return"],
            name="Buy & Hold (%)",
            line=dict(color="#888", width=1.8, dash="dash"),
            hovertemplate="B&H: %{y:.2f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="#aaa", width=1))

    fig.update_layout(
        title=dict(text="Kümülatif Getiri (%)", font=dict(size=15, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_feature_importance_chart(imp_df: pd.DataFrame) -> go.Figure:
    """Feature importance bar grafigi."""
    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
        marker=dict(
            color=imp_df["importance"],
            colorscale=[[0, "#E3F2FD"], [1, "#1976D2"]],
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Feature Importance (Top 10)", font=dict(size=15, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
    )
    return fig


def make_bollinger_chart(feature_df: pd.DataFrame) -> go.Figure:
    """Bollinger Bands grafigi."""
    fig = go.Figure()

    if "BB_Upper" in feature_df.columns:
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["BB_Upper"],
            name="Üst Bant", line=dict(color="#D32F2F", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["BB_Lower"],
            name="Alt Bant", line=dict(color="#2E7D32", width=1.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(25,118,210,0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=feature_df["Date"], y=feature_df["BB_Mid"],
            name="Orta Bant (MA20)", line=dict(color="#FB8C00", width=1.5, dash="dash"),
        ))

    fig.add_trace(go.Scatter(
        x=feature_df["Date"], y=feature_df["Close"],
        name="Kapanış", line=dict(color="#1976D2", width=2),
    ))

    fig.update_layout(
        title=dict(text="Bollinger Bands", font=dict(size=15, color="#1a1a2e")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "multi_pred" not in st.session_state:
    st.session_state.multi_pred = None
if "last_asset" not in st.session_state:
    st.session_state.last_asset = None
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0.0

# Sabitler
MIN_DAYS   = 180    # Model eğitimi için minimum gün sayısı
COOLDOWN_S = 30     # Butonlar arası minimum bekleme süresi (saniye)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 12px 0 8px 0;'>"
        "<span style='font-size:2rem;'>📈</span><br>"
        "<span style='font-size:1.1rem; font-weight:800; color:#1a1a2e;'>Algo Trading Bot</span><br>"
        "<span style='font-size:0.75rem; color:#888;'>ML Tabanlı Yön Tahmini</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Varlık Seçimi
    asset = st.selectbox(
        "📊 Varlık Seçin",
        ["BTC", "Altın", "USD/TRY"],
        index=0,
    )

    # Tarih Aralığı
    st.markdown("**📅 Tarih Aralığı**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input(
            "Başlangıç",
            value=date(2018, 1, 1),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
        )
    with col_e:
        end_date = st.date_input(
            "Bitiş",
            value=date.today(),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
        )

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # Tarih doğrulaması
    date_gap = (end_date - start_date).days
    date_error = None
    if start_date >= end_date:
        date_error = "⛔ Başlangıç tarihi bitiş tarihinden önce olmalıdır."
    elif date_gap < MIN_DAYS:
        date_error = f"⛔ En az {MIN_DAYS} günlük veri gereklidir. (Şu an: {date_gap} gün)"

    if date_error:
        st.markdown(
            f"<div style='background:#FFEBEE; border-left:4px solid #D32F2F; "
            f"border-radius:6px; padding:8px 12px; color:#B71C1C; font-size:0.82rem;'>"
            f"{date_error}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Cooldown kontrolü
    elapsed     = time.time() - st.session_state.last_run_time
    on_cooldown = elapsed < COOLDOWN_S
    btn_label   = (
        f"⏳ Lütfen {int(COOLDOWN_S - elapsed)}s bekleyin"
        if on_cooldown
        else "🚀 Tahmin Et & Analiz"
    )

    run_btn = st.button(
        btn_label,
        use_container_width=True,
        type="primary",
        disabled=bool(date_error) or on_cooldown,
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#aaa; text-align:center;'>"
        "⚠️ Bu uygulama yatırım tavsiyesi vermez.<br>"
        "Sadece eğitim amaçlıdır."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Ana Sayfa Başlığı
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='main-title'>📈 Algo Trading Bot</div>"
    "<div class='main-subtitle'>Altın · Döviz · Kripto — ML Tabanlı Yön Tahmini & Backtesting Dashboard</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='info-banner'>"
    "ℹ️ Sidebar'dan bir varlık seçip <strong>Tahmin Et & Analiz</strong> butonuna basın. "
    "Model ilk çalıştırmada eğitilecek, sonraki açılışlarda kaydedilen model yüklenecektir."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Buton Tıklandığında
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state.last_run_time = time.time()
    with st.spinner(f"⏳ {asset} verisi çekiliyor ve analiz yapılıyor..."):
        try:
            # Tam backtest pipeline — model otomatik seçilir
            result = run_full_backtest(
                asset=asset,
                start=start_str,
                model_type="auto",
                force_retrain=False,
            )
            st.session_state.result = result
            st.session_state.last_asset = asset

            # Sonraki gün tahmini — kazanan model zaten kaydedildi
            prediction = predict_next_day(
                asset=asset,
                start=start_str,
            )
            st.session_state.prediction = prediction

            # Son 7 gün çoklu tahmin
            multi_pred = predict_multi_day(
                asset=asset,
                n_days=7,
                start=start_str,
            )
            st.session_state.multi_pred = multi_pred

        except Exception as e:
            st.markdown(
                f"<div class='warn-banner'>❌ Hata oluştu: {e}</div>",
                unsafe_allow_html=True,
            )
            st.stop()


# ---------------------------------------------------------------------------
# Sonuçları Göster
# ---------------------------------------------------------------------------
if st.session_state.result is not None:
    result     = st.session_state.result
    prediction = st.session_state.prediction
    multi_pred = st.session_state.multi_pred
    curr_asset = st.session_state.last_asset

    backtest_df  = result["backtest_df"]
    bh_df        = result["bh_df"]
    metrics      = result["metrics"]
    feature_df   = result["feature_df"]
    raw_df       = result["raw_df"]
    feature_cols = result["feature_cols"]
    model        = result["model"]
    initial_cap  = result["initial_capital"]

    # -------------------------------------------------------------------
    # 1) Üst Metrikler
    # -------------------------------------------------------------------
    st.markdown("### 📊 Genel Bakış")
    m1, m2, m3, m4, m5 = st.columns(5)

    price_label = "Güncel Fiyat"
    price_val   = f"${prediction['price']:,.2f}" if prediction else "—"

    with m1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{price_label}</div>"
            f"<div class='value'>{price_val}</div>"
            f"<div class='delta' style='color:#888;'>{curr_asset}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        if prediction and prediction.get("try_gram"):
            tg = fmt_try_gram(prediction["try_gram"])
        else:
            tg = "—"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>TRY / gram</div>"
            f"<div class='value' style='font-size:1.2rem;'>{tg}</div>"
            f"<div class='delta' style='color:#888;'>Altın dönüşümü</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m3:
        tr = metrics["total_return_pct"]
        col = color_return(tr)
        sign = "+" if tr >= 0 else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Toplam Getiri</div>"
            f"<div class='value' style='color:{col};'>{sign}{tr:.1f}%</div>"
            f"<div class='delta' style='color:#888;'>Strateji</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m4:
        sh = metrics["sharpe_ratio"]
        sh_col = "#2E7D32" if sh > 1 else ("#FB8C00" if sh > 0 else "#D32F2F")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Sharpe Ratio</div>"
            f"<div class='value' style='color:{sh_col};'>{sh:.2f}</div>"
            f"<div class='delta' style='color:#888;'>> 1 iyi</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m5:
        md = metrics["max_drawdown_pct"]
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Max Drawdown</div>"
            f"<div class='value' style='color:#D32F2F;'>{md:.1f}%</div>"
            f"<div class='delta' style='color:#888;'>En büyük düşüş</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # 2) Tahmin Sinyali + Çoklu Gün Tahmini
    # -------------------------------------------------------------------
    st.markdown("### 🎯 Tahmin Sinyali")
    sig_col, multi_col = st.columns([1, 1.8])

    with sig_col:
        if prediction:
            sig = prediction["signal"]
            conf = prediction["confidence"]
            proba = prediction["proba_up"]
            sig_date = prediction["date"]

            css_class = {
                "AL": "signal-al",
                "SAT": "signal-sat",
                "BEKLE": "signal-bekle",
            }.get(sig, "signal-bekle")

            color = prediction["color"]
            emoji = prediction["emoji"]

            st.markdown(
                f"<div class='{css_class}'>"
                f"<div class='signal-text' style='color:{color};'>{emoji} {sig}</div>"
                f"<div class='signal-sub'>Güven: <strong>%{conf:.1f}</strong></div>"
                f"<div class='signal-sub'>Yukarı olasılığı: <strong>{proba:.1%}</strong></div>"
                f"<div class='signal-sub' style='margin-top:8px; font-size:0.78rem;'>📅 {sig_date}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Güven gauge
            st.markdown("<br>", unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Yukarı Olasılığı", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 45], "color": "#FFEBEE"},
                        {"range": [45, 55], "color": "#FFF8E1"},
                        {"range": [55, 100], "color": "#E3F2FD"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a1a2e", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                number={"suffix": "%", "font": {"size": 28}},
            ))
            gauge.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="white",
            )
            st.plotly_chart(gauge, use_container_width=True)

    with multi_col:
        st.markdown("**📅 Son 7 Günün Tahmin Tablosu**")
        if multi_pred:
            rows_html = ""
            for row in reversed(multi_pred):
                sig_color = {"AL": "#1976D2", "SAT": "#D32F2F", "BEKLE": "#F57C00"}.get(row["signal"], "#888")
                close_fmt = f"${row['close']:,.4f}" if row["close"] else "—"
                rows_html += (
                    f"<tr>"
                    f"<td>{row['date']}</td>"
                    f"<td>{close_fmt}</td>"
                    f"<td style='color:{sig_color}; font-weight:700;'>{row['emoji']} {row['signal']}</td>"
                    f"<td>{row['proba_up']:.1%}</td>"
                    f"<td>%{row['confidence']:.1f}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"<table class='signal-table' style='width:100%; border-collapse:collapse;'>"
                f"<thead><tr>"
                f"<th>Tarih</th><th>Kapanış</th><th>Sinyal</th><th>↑ Olasılık</th><th>Güven</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

        # Uyarı notu
        st.markdown(
            "<div class='warn-banner' style='margin-top:12px;'>"
            "⚠️ Bu tahminler geçmiş veriye dayanır. Finansal yatırım kararları için kullanmayın."
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------
    # 3) Fiyat Grafigi
    # -------------------------------------------------------------------
    st.markdown("### 📉 Fiyat & Hareketli Ortalamalar")

    # Görüntülenecek periyot seçici
    period_options = {"Tüm Veri": None, "Son 1 Yıl": 252, "Son 6 Ay": 126, "Son 3 Ay": 63, "Son 1 Ay": 21}
    period_sel = st.select_slider(
        "Görüntülenecek Periyot",
        options=list(period_options.keys()),
        value="Son 1 Yıl",
    )
    n_rows = period_options[period_sel]
    display_df = feature_df.tail(n_rows) if n_rows else feature_df

    st.plotly_chart(make_price_chart(display_df, curr_asset), use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------
    # 4) RSI + Momentum
    # -------------------------------------------------------------------
    st.markdown("### 📡 Teknik İndikatörler")
    ind_col1, ind_col2 = st.columns(2)
    with ind_col1:
        if "RSI_14" in display_df.columns:
            st.plotly_chart(make_rsi_chart(display_df), use_container_width=True)
        else:
            st.info("RSI verisi bulunamadı.")
    with ind_col2:
        if "Momentum" in display_df.columns:
            st.plotly_chart(make_momentum_chart(display_df), use_container_width=True)
        else:
            st.info("Momentum verisi bulunamadı.")

    # Bollinger Bands
    with st.expander("📊 Bollinger Bands Göster"):
        if "BB_Upper" in display_df.columns:
            st.plotly_chart(make_bollinger_chart(display_df), use_container_width=True)
        else:
            st.info("Bollinger Bands verisi bulunamadı.")

    st.divider()

    # -------------------------------------------------------------------
    # 5) Backtest
    # -------------------------------------------------------------------
    st.markdown("### 💰 Backtest Simülasyonu")

    currency_sym = "₺" if curr_asset == "Altın" else "$"

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Başlangıç Sermayesi</div>"
            f"<div class='value'>{currency_sym}{initial_cap:,.0f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with bc2:
        fv = metrics["final_value"]
        fv_col = color_return(fv - initial_cap)
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Final Değer</div>"
            f"<div class='value' style='color:{fv_col};'>{currency_sym}{fv:,.0f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with bc3:
        wr = metrics["win_rate_pct"]
        wr_col = "#2E7D32" if wr >= 50 else "#D32F2F"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Kazanma Oranı</div>"
            f"<div class='value' style='color:{wr_col};'>%{wr:.1f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with bc4:
        bh_ret = metrics["buy_hold_return_pct"]
        bh_col = color_return(bh_ret)
        sign = "+" if bh_ret >= 0 else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Buy & Hold Getiri</div>"
            f"<div class='value' style='color:{bh_col};'>{sign}{bh_ret:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Backtest portföy grafiği
    st.plotly_chart(make_backtest_chart(backtest_df, bh_df, initial_cap), use_container_width=True)

    # Kümülatif getiri
    st.plotly_chart(make_cumulative_return_chart(backtest_df, bh_df), use_container_width=True)

    # Detaylı metrikler
    with st.expander("📋 Detaylı Backtest Metrikleri"):
        col_m1, col_m2 = st.columns(2)
        metric_items = list(metrics.items())
        mid = len(metric_items) // 2
        with col_m1:
            for k, v in metric_items[:mid]:
                label = k.replace("_", " ").title()
                st.markdown(f"**{label}:** `{v}`")
        with col_m2:
            for k, v in metric_items[mid:]:
                label = k.replace("_", " ").title()
                st.markdown(f"**{label}:** `{v}`")

    st.divider()

    st.markdown(
        "<div class='warn-banner'>"
        "⚠️ Geçmiş performans gelecek getiriyi garanti etmez."
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # Henüz analiz yapılmamış — hoş geldin ekranı
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            "<div style='text-align:center; padding: 40px 30px; background:#f8f9ff; "
            "border-radius:16px; border: 1.5px solid #e0e4f0;'>"
            "<div style='font-size:4rem;'>📈</div>"
            "<div style='font-size:1.4rem; font-weight:800; color:#1a1a2e; margin-top:12px;'>"
            "Başlamak için hazır!</div>"
            "<div style='color:#666; margin-top:8px; font-size:0.95rem;'>"
            "Soldan bir <strong>varlık seçin</strong> ve "
            "<strong>Tahmin Et & Analiz</strong> butonuna basın.</div>"
            "<div style='margin-top:20px; display:flex; gap:12px; justify-content:center; "
            "flex-wrap:wrap;'>"
            "<span style='background:#E3F2FD; color:#1976D2; padding:6px 16px; "
            "border-radius:20px; font-weight:600; font-size:0.85rem;'>₿ BTC/USD</span>"
            "<span style='background:#FFF8E1; color:#F57C00; padding:6px 16px; "
            "border-radius:20px; font-weight:600; font-size:0.85rem;'>🥇 Altın XAU/USD</span>"
            "<span style='background:#FFEBEE; color:#D32F2F; padding:6px 16px; "
            "border-radius:20px; font-weight:600; font-size:0.85rem;'>💱 USD/TRY</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer'>"
    "Algo Trading Bot · ML Tabanlı Yön Tahmini · "
    "Veriler: <strong>Yahoo Finance</strong> · "
    "Bu uygulama yatırım tavsiyesi vermez."
    "</div>",
    unsafe_allow_html=True,
)
