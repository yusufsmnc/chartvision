# 📈 ChartVision — Multi-Modal Stock Movement Prediction

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Multi-modal deep learning system combining **candlestick chart images** (CNN) and **technical indicators** (LSTM) for stock movement prediction.

## 🚀 Live Demo
[Coming soon — Streamlit Cloud]

## 📊 Results

| Model | Val Accuracy |
|---|---|
| LSTM (Time Series only) | 43.29% |
| CNN (Chart Images only) | 52.70% |
| **Multi-Modal (CNN + LSTM)** | **53.07%** |

| Ticker | Sharpe Ratio | Strategy Return | Buy & Hold |
|---|---|---|---|
| AAPL | -0.27 | -0.8% | +30.6% |
| MSFT | 1.64 | +87.5% | -5.6% |
| GOOGL | 2.18 | +85.8% | +10.3% |
| TSLA | 0.42 | — | — |

## 🏗️ Architecture

```
Candlestick Chart (PNG)
↓
ResNet18 (CNN)
↓
128-dim embedding
↓                    → Fusion (272-dim) → 3-class output
128-dim embedding
↑
LSTM (2 layers)
↑
Technical Indicators
(RSI, MACD, Bollinger Bands,
Volume SMA, OHLCV)
↑
Ticker Embedding (16-dim)
```

## 📁 Project Structure

```
chartvision/
├── app.py                  # Streamlit demo
├── src/
│   └── model.py            # Model mimarisi
├── models/
│   ├── best_multimodal_v3.pth
│   └── scaler.pkl
├── requirements.txt
├── README.md
└── DEVELOPMENT.md          # Geliştirme süreci ve iyileştirme planı
```

## 🛠️ Tech Stack

- **Model:** PyTorch, ResNet18, LSTM, Ticker Embedding
- **Data:** yfinance, mplfinance, ta (technical analysis)
- **Demo:** Streamlit
- **Training:** Google Colab (T4 GPU)

## 📈 Dataset

- **Hisseler:** AAPL, MSFT, GOOGL, TSLA
- **Dönem:** 2019-2024 (6 yıl)
- **Toplam örnek:** 5,664
- **Train/Val/Test:** 70% / 15% / 15% (zamansal split)
- **Görüntü sayısı:** 5,796 mum grafiği PNG

## 🔍 How It Works

1. Son 60 günlük OHLCV verisi indirilir
2. Mum grafiği PNG olarak üretilir → ResNet18'e verilir
3. Teknik indikatörler hesaplanır → LSTM'e verilir
4. Ticker embedding eklenir
5. CNN + LSTM + Embedding birleştirilir
6. 3 sınıf tahmin üretilir: ⬆️ Yukarı / ➡️ Yatay / ⬇️ Aşağı

## ⚠️ Disclaimer

Bu proje bir **ML araştırma projesidir**. Yatırım tavsiyesi değildir.
Backtesting sonuçları gerçek trading performansını yansıtmaz
(transaction cost, slippage hesaba katılmamıştır).

## 👤 Author

[yusufsmnc](https://github.com/yusufsmnc)