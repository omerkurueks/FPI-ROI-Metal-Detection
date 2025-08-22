import os
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
from ultralytics import YOLOE
import torch

# CUDA uygunluk kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# HuggingFace'den ağırlık indir
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt", local_dir='.')

# --- SABİT ROI KOORDINATLARI ---
# İnteraktif olarak seçilen ROI koordinatları
FIXED_ROI = [235, 1010, 780, 1241]  # [x1, y1, x2, y2]

print(f"🎯 Sabit ROI koordinatları: {FIXED_ROI}")

# --- Metal tespit parametreleri ---
IMAGE_PATH = "data/fpi.jpeg"
NAMES = ["metal bar", "steel bar", "iron bar", "rectangular prism", "metal block", "steel beam", "iron beam", "metal piece", "industrial part", "steel product"]

def filter_detections_by_roi(detections, roi_bbox):
    """ROI içerisindeki tespitleri filtrele"""
    x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
    
    filtered_indices = []
    for i, bbox in enumerate(detections.xyxy):
        x1, y1, x2, y2 = bbox
        
        # Bbox'ın merkez noktası ROI içinde mi kontrol et
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if x1_roi <= center_x <= x2_roi and y1_roi <= center_y <= y2_roi:
            filtered_indices.append(i)
    
    # Filtrelenmiş tespitleri döndür
    if filtered_indices:
        return detections[filtered_indices]
    else:
        return sv.Detections.empty()

def draw_roi_on_image(image, roi_bbox, color=(255, 0, 0), thickness=4):
    """ROI alanını görüntü üzerinde çiz"""
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = roi_bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    
    # ROI bilgilerini yazdır
    roi_width = x2 - x1
    roi_height = y2 - y1
    text = f"ROI: {roi_width}x{roi_height}px"
    draw.text((x1, y1-25), text, fill=color)
    
    return image

def detect_metals_in_fixed_roi():
    """Sabit ROI ile metal tespiti"""
    
    # Model yükle
    print("🤖 Model yükleniyor...")
    model = YOLOE("yoloe-v8l-seg.pt")
    if device == "cuda":
        model = model.cuda()
    model.set_classes(NAMES, model.get_text_pe(NAMES))
    
    # Görüntü yükle ve tahmin yap
    print("🔍 Metal tespiti yapılıyor...")
    image = Image.open(IMAGE_PATH)
    results = model.predict(image, conf=0.1, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    
    print(f"Toplam tespit: {len(detections)}")
    
    # ROI içerisindeki tespitleri filtrele
    roi_detections = filter_detections_by_roi(detections, FIXED_ROI)
    print(f"ROI içerisindeki tespit: {len(roi_detections)}")
    
    # Sonuçları görselleştir
    print("🎨 Sonuçlar görselleştiriliyor...")
    annotated_image = image.copy()
    
    # ROI alanını çiz (kırmızı dikdörtgen)
    annotated_image = draw_roi_on_image(annotated_image, FIXED_ROI, color=(255, 0, 0), thickness=4)
    
    # Tüm tespitleri gri renkte çiz
    if len(detections) > 0:
        annotated_image = sv.BoxAnnotator(color=sv.Color.from_hex("#808080")).annotate(
            scene=annotated_image, detections=detections)
    
    # ROI içerisindeki tespitleri yeşil renkte çiz
    if len(roi_detections) > 0:
        annotated_image = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=4).annotate(
            scene=annotated_image, detections=roi_detections)
        annotated_image = sv.LabelAnnotator(color=sv.Color.from_hex("#00FF00")).annotate(
            scene=annotated_image, detections=roi_detections)
    
    # Sonucu kaydet
    output_path = "fixed_roi_metal_detection.jpg"
    annotated_image.save(output_path)
    
    # Detaylı rapor
    print("\n" + "="*60)
    print("📊 SABİT ROI METAL TESPİT RAPORU")
    print("="*60)
    print(f"📁 Görüntü: {IMAGE_PATH}")
    print(f"🎯 ROI Koordinatları: {FIXED_ROI}")
    print(f"📏 ROI Boyutu: {FIXED_ROI[2]-FIXED_ROI[0]}x{FIXED_ROI[3]-FIXED_ROI[1]} piksel")
    print(f"🔍 Toplam metal tespit: {len(detections)}")
    print(f"✅ ROI içerisindeki metal: {len(roi_detections)}")
    print(f"📁 Sonuç dosyası: {output_path}")
    print("-" * 60)
    
    if len(roi_detections) > 0:
        print(f"🟢 ROI içerisindeki metal nesneler:")
        for i, bbox in enumerate(roi_detections.xyxy):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"   Nesne {i+1}:")
            print(f"     • Konum: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
            print(f"     • Boyut: {width:.0f}x{height:.0f} piksel")
            print(f"     • Merkez: ({center_x:.0f}, {center_y:.0f})")
    else:
        print(f"❌ Seçilen ROI alanında metal nesne bulunamadı.")
    
    print("="*60)
    print("🔴 Kırmızı dikdörtgen: Sabit ROI alanı")
    print("🟢 Yeşil kutular: ROI içerisindeki metal nesneler")
    print("⚫ Gri kutular: ROI dışındaki tespitler")
    
    return len(roi_detections), output_path

# ROI koordinatlarını değiştirmek için bu değişkeni düzenleyin:
def update_roi(new_coordinates):
    """ROI koordinatlarını güncelle"""
    global FIXED_ROI
    FIXED_ROI = new_coordinates
    print(f"🔄 ROI koordinatları güncellendi: {FIXED_ROI}")

if __name__ == "__main__":
    print("🚀 Sabit ROI ile Metal Tespiti Başlatılıyor...")
    print(f"📍 Mevcut ROI: {FIXED_ROI}")
    print("-" * 60)
    
    metal_count, result_file = detect_metals_in_fixed_roi()
    
    print(f"\n🎉 İşlem tamamlandı!")
    print(f"📊 Sonuç: ROI içerisinde {metal_count} adet metal tespit edildi")
    print(f"🖼️  Sonuç dosyası: {result_file}")

