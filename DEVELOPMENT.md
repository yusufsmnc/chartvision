# 🔬 Development Notes — ChartVision

Bu dosya projenin mimari kararlarını, geliştirme sürecini,
tespit edilen problemleri ve iyileştirme stratejilerini içerir.

---

## 📅 Geliştirme Süreci

### Veri Pipeline
- AAPL, MSFT, GOOGL, TSLA için 2019-2024 arası OHLCV verisi çekildi
- Sliding window (60 gün) ile mum grafiği PNG'leri üretildi
- RSI, MACD, Bollinger Bands, Volume SMA teknik indikatörleri hesaplandı
- 3-class etiket stratejisi uygulandı (%1 eşik)
- Her hisse için ayrı zamansal split yapıldı (70/15/15)
- Toplam 5,796 görüntü, 5,664 örnek

### Model Geliştirme
- CNN baseline (ResNet18) ile başlandı → Val Acc %52
- LSTM modeli eklendi → tek başına Val Acc %43
- CNN + LSTM fusion denendi → Val Acc %53
- Ticker embedding eklendi → Val Acc %53.07
- Backtesting yapıldı → MSFT Sharpe 1.64, GOOGL Sharpe 2.18

### Deployment
- Streamlit demo yazıldı ve lokalde test edildi
- GitHub'a push edildi
- Streamlit Cloud deploy planlanıyor

---

## 🏗️ Mimari Kararlar

### Neden ResNet18?
- ImageNet pretrained ağırlıkları mevcut — transfer learning için ideal
- Hafif ve hızlı — az GPU ile eğitilebilir
- 224x224 görüntülerle iyi çalışıyor
- Baseline için yeterince güçlü
- Alternatif: ViT (Vision Transformer) — daha iyi sonuç verebilir ama daha ağır

### Neden LSTM?
- Zaman serisi verisi için tasarlanmış
- 60 günlük bağımlılıkları öğrenebilir
- Teknik indikatörlerin zamansal ilişkisini yakalar
- Alternatif: Temporal Fusion Transformer — daha iyi ama çok daha karmaşık

### Neden Ticker Embedding?
- Her hissenin karakteri farklı (TSLA volatil, GOOGL stabil)
- Model hisse kimliğini öğrenebilir
- Sadece 16 boyut — çok az parametre ekliyor
- Sonuç: %52.83 → %53.07 (küçük iyileşme)

### Neden 3-class?
- Binary (yukarı/aşağı) çok gürültülü
- %1 eşiği ile yatay bölge tanımlandı
- Belirsiz günleri ayrı sınıfa koymak modeli rahatlatıyor
- Sorun: yatay sınıf çok dominant (%51)

### Neden Zamansal Split?
- Rastgele split → look-ahead bias (gelecek veriyi görme)
- Her hisse için ayrı split → cross-asset sızıntısı önlendi
- Gerçek trading koşullarını simüle ediyor
---

## ⚠️ Tespit Edilen Problemler

### 1. Sınıf Dengesizliği
- Yatay: %51, Yukarı: %28, Aşağı: %21
- Model neredeyse her şeyi yatay tahmin ediyor
- F1-score aşağı ve yukarı sınıfta çok düşük

### 2. Overfitting
- Train accuracy %100'e ulaşıyor
- Val accuracy %53'te kalıyor
- Az veri (5,664 örnek) + derin model = ezberleme
- Dropout ve early stopping kısmen çözdü ama yetmedi


### 3. Piyasa Rejimi Değişimi
- Train dönemi (2019-2022): güçlü bull market
- Test dönemi (2023-2024): farklı koşullar
- AAPL val Acc %68 iken test Acc %25'e düştü
- Model farklı piyasa koşullarına genelleme yapamıyor

### 4. Cross-Asset Genelleme
- AAPL tek başına %68, 4 hisse birlikte %53
- Her hissenin volatilite ve momentum karakteri farklı
- TSLA için %1 eşik yetersiz — çok daha volatil

### 5. Veri Miktarı
- 4 hisse, 5 yıl → 5,664 örnek
- Derin öğrenme için yetersiz
- Daha fazla hisse ve daha uzun dönem gerekiyor


---

## 🚀 İyileştirme Stratejileri

### Kısa Vadeli

**Sınıf Dengesizliği:** CrossEntropyLoss'a sınıf ağırlıkları eklenerek az olan sınıflara daha fazla ağırlık verilebilir. Aşağı sınıfına 2.0, yatay sınıfına 0.5, yukarı sınıfına 1.5 ağırlık uygulanabilir.

**Eşik Değişikliği:** %1 yerine %2 eşik kullanılarak daha net sinyal elde edilebilir. Bu değişiklik yatay örneklerin sayısını azaltır ve sınıf dengesizliğini iyileştirir.

**Focal Loss:** Zor örneklere daha fazla ağırlık veren Focal Loss kullanılarak modelin daha zor sınıfları öğrenmesi sağlanabilir.

### Orta Vadeli

**Daha Fazla Veri:**
- S&P 500'den 20-30 hisse ekle
- 10 yıllık veri kullan (2014-2024)
- Farklı sektörler: finans, enerji, sağlık

**Daha İyi Feature'lar:**
- Haber sentiment skoru (FinBERT)
- VIX — piyasa korku endeksi
- Sektör ETF korelasyonu
- 52 haftalık high/low mesafesi

**Vision Transformer:** ResNet18 yerine ViT kullanılarak grafik pattern tespiti iyileştirilebilir.

### Uzun Vadeli

**Temporal Fusion Transformer:**
- LSTM yerine state-of-the-art zaman serisi modeli
- Attention mechanism ile önemli günlere odaklanır
- pytorch-forecasting kütüphanesi ile uygulanabilir

**Piyasa Rejimi Tespiti:**
- Önce rejimi tespit et (bull/bear/sideways)
- Her rejim için ayrı model eğit
- HMM (Hidden Markov Model) ile rejim tespiti

**Walk-Forward Validation:**
- Sabit train/val/test yerine kayan pencere
- Gerçek trading koşullarına çok daha yakın
- Her ay model yeniden eğitilir
---

## 📊 Model Karşılaştırma

| Deney | Val Acc | Test Acc | Not |
|---|---|---|---|
| CNN — AAPL tek | %69 | — | Overfitting var |
| CNN — 4 hisse | %52 | — | Data augmentation eklendi |
| LSTM — 4 hisse | %43 | — | Tek başına zayıf |
| Fusion — AAPL tek | %68.87 | — | En iyi single-asset sonuç |
| Fusion — yanlış split | %39 | — | Split hatası |
| Fusion — doğru split | %52.83 | — | Her hisse ayrı split |
| **Fusion + Embedding** | **%53.07** | **%49.18** | **Final model** |

---
## 🧪 Sınıf Dengesizliği Deneyleri

Sınıf dengesizliği problemini çözmek için üç farklı yaklaşım denendi.

### Deney 1 — Eşik Artırma (%1 → %2)
Yatay bölgeyi daraltmak yerine büyüttü. Sonuç: Yatay %71'e çıktı (önceden %51). Beklenenin tersi etki yarattı, bu yaklaşımdan vazgeçildi.

### Deney 2 — Eşik Azaltma (%1 → %0.5)
Sınıflar daha dengeli hale geldi (Yukarı %41, Aşağı %34, Yatay %24) ancak günlük piyasa gürültüsü sinyal olarak algılandı. Val Acc %37.74'e düştü — orijinal %53.07'nin çok altında. Küçük eşik, sinyal/gürültü oranını bozdu.

### Deney 3 — Sınıf Ağırlıkları (orijinal %1 eşik ile)
CrossEntropyLoss'a sınıf ağırlıkları eklendi. Val Acc %51.89 — orijinal modele yakın ama daha düşük. Ağırlıklandırma modelin az örnekli sınıflara aşırı odaklanmasına, bu da genelleme kabiliyetinin azalmasına yol açtı.

### Sonuç
Üç deney de orijinal modeli (%53.07) geçemedi. Sınıf dengesizliği bu veri setinin doğal bir özelliği — piyasa çoğunlukla yatay hareket ediyor. Zorla dengelemek yerine daha fazla veri ve daha iyi feature'lar ile ilerlemek daha etkili bulundu.

| Deney | Val Acc | Sonuç |
|---|---|---|
| Orijinal (%1 eşik, embedding) | %53.07 | Baseline |
| Eşik %2 | — | Denenmeden vazgeçildi |
| Eşik %0.5 + ağırlık | %37.74 | Başarısız |
| Eşik %1 + ağırlık | %51.89 | Hafif düşüş |
---

## 📈 Veri Genişletme Deneyi (16 Hisse)

Overfitting ve genelleme problemini çözmek için veri seti 4 hisseden 16 hisseye çıkarıldı.

### Eklenen Hisseler
Farklı sektörlerden 12 yeni hisse eklendi:
- **Tech:** NVDA, META, AMZN, NFLX
- **Finans:** JPM, BAC
- **Enerji:** XOM, CVX
- **Sağlık:** JNJ, PFE
- **Tüketici:** WMT, DIS

### Veri Seti Büyümesi
| Metrik | Önce | Sonra |
|---|---|---|
| Hisse sayısı | 4 | 16 |
| Toplam örnek | 5,664 | 22,656 |
| Görüntü sayısı | 5,796 | 23,184 |
| Train/Val/Test | 3964/848/852 | 15856/3392/3408 |

### Sonuçlar
| Model | Val Acc | Train Acc | Not |
|---|---|---|---|
| 4 hisse | %53.07 | %100 | Aşırı overfitting |
| **16 hisse + weight decay** | **%56.25** | %65 | Overfitting azaldı |
| 16 hisse + sınıf ağırlığı | Macro F1 %37.86 | — | Başarısız |

### Öne Çıkan Bulgular
- **Daha fazla veri gerçek iyileşme sağladı** (%53 → %56)
- **Overfitting ciddi şekilde azaldı** — train accuracy %100'den %65'e düştü, model artık ezberlemiyor
- **Sınıf ağırlıkları yine başarısız oldu** — 16 hisse ile de faydası olmadı

### Kalıcı Problem: Sınıf Dengesizliği
Test seti classification report'u modelin hâlâ çoğunlukla "Yatay" tahmin ettiğini gösterdi:
- Aşağı recall: 0.00 (neredeyse hiç aşağı tahmin yok)
- Yatay recall: 0.95 (her şeyi yatay diyor)
- Yukarı recall: 0.13

Bu, finansal ML'de bilinen bir olgudur — piyasa büyük ölçüde tahmin edilemez olduğu için model belirsizlikte "yatay" demeyi öğrenir. Sınıf ağırlıkları bu davranışı zorla değiştirmeye çalıştığında, doğru olan yatay tahminler de kaçırıldığı için toplam performans düşer.