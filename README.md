# FPI ROI Metal Detection

Bu proje, IP kameralardan RTSP protokolü ile canlı görüntü alarak, belirlenen ROI (Region of Interest) alanında metal ve diğer nesnelerin tespitini yapmaktadır.

## Özellikler

- **RTSP IP Kamera Desteği**: IP kameralardan canlı görüntü alma
- **ROI Tabanlı Tespit**: Belirli alan içerisinde nesne tespiti
- **YOLOE Modeli**: Text-guided object detection
- **CUDA Desteği**: GPU ile hızlandırılmış işlem
- **Gerçek Zamanlı Görselleştirme**: Canlı tespit sonuçları

## Tespit Edilen Nesneler

### Metal Nesneler
- Metal bar, steel bar, iron bar
- Metal block, steel beam, iron beam
- Metal piece, industrial part, steel product

### Günlük Eşyalar
- Hand (el)
- Pen, pencil (kalem)
- Notebook, book (defter, kitap)
- Cup, mug, glass (bardak, kupa)
- Bottle, water bottle (şişe)

## Kurulum

1. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Model dosyalarını indirin:**
- `yoloe-v8l-seg.pt`
- `mobileclip_blt.pt`

## Kullanım

### Statik Görüntü Analizi
```bash
cd src
python main.py
```

### Canlı Kamera Analizi
```bash
cd src
python cam.py
```

## Dosya Yapısı

```
FPI-ROI-Metal-Detection/
├── src/
│   ├── main.py          # Statik görüntü analizi
│   └── cam.py           # Canlı kamera analizi
├── data/
│   └── fpi.jpeg         # Test görüntüsü
├── output/              # Sonuç dosyaları
├── models/              # Model dosyaları
├── requirements.txt     # Python bağımlılıkları
└── README.md           # Bu dosya
```

## Konfigürasyon

### ROI Koordinatları
`src/cam.py` ve `src/main.py` dosyalarında ROI koordinatlarını değiştirebilirsiniz:
```python
FIXED_ROI = [235, 1010, 780, 1241]  # [x1, y1, x2, y2]
```

### RTSP Kamera Adresi
`src/cam.py` dosyasında RTSP adresini güncelleyin:
```python
rtsp_url = "rtsp://kullanici:sifre@ip:port/stream"
```

## Gereksinimler

- Python 3.8+
- CUDA (opsiyonel, GPU desteği için)
- IP kamera (RTSP desteği ile)

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## İletişim

Proje Sahibi - [@OmerKuru](https://github.com/OmerKuru)

Proje Linki: [https://github.com/OmerKuru/FPI-ROI-Metal-Detection](https://github.com/OmerKuru/FPI-ROI-Metal-Detection)

Fabrika ortamında üretilen dikdörtgen prizma demir parçalarını belirli bir ROI (Region of Interest) alanında tespit etmek için geliştirilmiş AI tabanlı sistem.

##  Özellikler

- **CUDA Destekli GPU Hızlandırması**: RTX/GTX serisi GPU'larda optimize edilmiş performans
- **Zero-Shot Segmentasyon**: YOLOE modeli ile metin tabanlı nesne tespiti
- **Sabit ROI Sistemi**: Önceden tanımlanmış alanda hassas tespit
- **Detaylı Raporlama**: Görsel ve metin tabanlı sonuç analizi
- **Endüstriyel Kullanım**: Fabrika ortamı için optimize edilmiş

##  Proje Yapısı

```
FPI-ROI-Metal-Detection/
 src/                    # Kaynak kodları
    main.py            # Ana tespit sistemi
 data/                  # Görüntü dosyaları
    fpi.jpeg          # Test görüntüsü
 models/               # AI model dosyaları (otomatik indirilir)
 output/               # Sonuç görüntüleri ve raporlar
 docs/                 # Dokümantasyon
 requirements.txt      # Python paket gereksinimleri
 README.md            # Bu dosya
```

##  Kurulum

### 1. Gereksinimler
- Python 3.8+
- CUDA destekli GPU (önerilen)
- 8GB+ RAM

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Test Görüntüsünü Ekleyin
`fpi.jpeg` dosyasını `data/` klasörüne koyun.

##  Kullanım

### Basit Kullanım
```bash
python src/main.py
```

### Python Kodu
```python
from src.main import FPIMetalDetector

# Sistem oluştur
detector = FPIMetalDetector()

# ROI koordinatlarını özelleştir (isteğe bağlı)
detector.roi = [235, 1010, 780, 1241]  # [x1, y1, x2, y2]

# Tespit yap
result = detector.detect("data/fpi.jpeg")

# Raporu yazdır
detector.print_report(result)
```

##  Yapılandırma

### ROI Koordinatlarını Değiştirme
`src/main.py` dosyasında `FPIMetalDetector` sınıfında:
```python
self.roi = [x1, y1, x2, y2]  # Yeni koordinatlar
```

### Tespit Parametreleri
```python
detector = FPIMetalDetector(
    roi_coordinates=[235, 1010, 780, 1241],
    confidence_threshold=0.1
)
```

##  Çıktı Formatı

### Görsel Sonuç
-  Kırmızı dikdörtgen: ROI alanı
-  Yeşil kutular: ROI içerisindeki metal nesneler
-  Gri kutular: ROI dışındaki tespitler

### Konsol Raporu
```
 FPI METAL TESPİT RAPORU
============================================================
 Görüntü: fpi.jpeg
 ROI: [235, 1010, 780, 1241]
 ROI Boyutu: 545x231 piksel
 Toplam metal: 4
 ROI içerisinde: 1
```

##  Teknik Detaylar

### AI Modeli
- **YOLOE**: Zero-shot object detection
- **MobileCLIP**: Text-to-image understanding
- **Backbone**: Vision Transformer

### Desteklenen Metal Türleri
- Metal bar, Steel bar, Iron bar
- Rectangular prism, Metal block
- Steel beam, Iron beam
- Industrial part, Steel product

##  Sorun Giderme

### CUDA Hatası
```bash
# CPU modunda çalıştırın
export CUDA_VISIBLE_DEVICES=""
python src/main.py
```

### Model İndirme Sorunu
Model dosyaları otomatik olarak HuggingFace'den indirilir. İnternet bağlantınızı kontrol edin.

### Bellek Hatası
Görüntü boyutunu küçültün veya batch size'ı azaltın.

##  Performans

- **GPU (RTX 3060)**: ~2-3 saniye/görüntü
- **CPU**: ~10-15 saniye/görüntü
- **Bellek Kullanımı**: ~2GB VRAM, ~4GB RAM

##  Lisans

Bu proje Ar-Ge amaçlı geliştirilmiştir.

##  Katkıda Bulunanlar

- [@OmerKuru] - Ana geliştirici

##  İletişim

Proje hakkında sorularınız için issue açabilirsiniz.
