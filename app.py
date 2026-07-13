import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import ta
import pickle
import tempfile
import os
import sys

sys.path.append(os.path.dirname(__file__))
from src.model import MultiModalModelV2

# 16 hisse
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "JPM",
           "BAC", "XOM", "CVX", "JNJ", "PFE", "WMT", "DIS", "NFLX"]
ticker2id = {t: i for i, t in enumerate(tickers)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = MultiModalModelV2(num_tickers=16)
    model.load_state_dict(torch.load(
        "models/best_multimodal_16.pth",
        map_location="cpu"
    ))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    with open("models/scaler_16.pkl", "rb") as f:
        return pickle.load(f)

def predict(ticker, df, scaler, model):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    mpf.plot(df.iloc[-60:], type="candle", style="charles", savefig=tmp_path)
    img = Image.open(tmp_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal()
    df["bb_high"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["bb_low"] = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
    df["volume_sma"] = df["Volume"].rolling(20).mean()
    df = df.dropna()

    window = df.iloc[-60:][["Close", "Volume", "rsi", "macd",
                             "macd_signal", "bb_high", "bb_low",
                             "volume_sma"]].values
    window_scaled = scaler.transform(window.reshape(-1, 8)).reshape(1, 60, 8)
    ts_tensor = torch.FloatTensor(window_scaled)
    tid_tensor = torch.LongTensor([ticker2id[ticker]])

    with torch.no_grad():
        output = model(img_tensor, ts_tensor, tid_tensor)
        pred = output.argmax(1).item()
        probs = torch.softmax(output, dim=1)[0].numpy()

    labels = {0: "⬇️ Aşağı", 1: "➡️ Yatay", 2: "⬆️ Yukarı"}
    return labels[pred], probs, img

# UI
st.title("📈 ChartVision — Hisse Tahmin")
st.write("Multi-modal deep learning ile hisse senedi yön tahmini (16 hisse)")

ticker = st.selectbox("Hisse seç", tickers)

if st.button("Tahmin Et"):
    with st.spinner("Veri indiriliyor..."):
        df = yf.download(ticker, period="1y")
        df.columns = df.columns.droplevel(1)

    model = load_model()
    scaler = load_scaler()

    with st.spinner("Tahmin yapılıyor..."):
        label, probs, chart_img = predict(ticker, df, scaler, model)

    st.subheader(f"Tahmin: {label}")

    confidence = probs.max()
    THRESHOLD = 0.45

    col1, col2, col3 = st.columns(3)
    col1.metric("⬇️ Aşağı", f"{probs[0]:.1%}")
    col2.metric("➡️ Yatay", f"{probs[1]:.1%}")
    col3.metric("⬆️ Yukarı", f"{probs[2]:.1%}")

    if confidence < THRESHOLD:
        st.info(f"🤔 **Model emin değil** (güven: {confidence:.1%}). "
                f"Selective prediction eşiğinin ({THRESHOLD:.0%}) altında — "
                f"bu tahmin düşük güvenilirlikte.")
    else:
        st.success(f"✅ Model bu tahminde görece emin (güven: {confidence:.1%})")

    st.image(chart_img, caption="Son 60 günlük mum grafiği")

    st.warning(
        "⚠️ **Bu bir ML araştırma projesidir, yatırım tavsiyesi değildir.**\n\n"
        "Modelin test accuracy'si %55.8. Ancak backtesting negatif Sharpe ratio "
        "(-0.10) veriyor ve model 'Aşağı' sınıfını neredeyse hiç tahmin etmiyor. "
        "Detaylı analiz için repo'daki DEVELOPMENT.md'ye bakın."
    )