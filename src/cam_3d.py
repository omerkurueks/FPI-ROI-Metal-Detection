import cv2
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
from ultralytics import YOLOE
import torch
from datetime import datetime
import os

# 3D detection imports
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator

# Sabit ROI koordinatları (1920x1080 için merkezi ROI)
FIXED_ROI = [400, 200, 1520, 880]  # [x1, y1, x2, y2] - merkezi büyük ROI
NAMES = [
    "metal bar", "steel bar", "iron bar", "rectangular prism", "metal block",
    "steel beam", "iron beam", "metal piece", "industrial part", "steel product",
    "hand", "pen", "pencil", "notebook", "book", "cup", "mug", "glass", 
    "bottle", "water bottle", "beverage bottle"
]

class ROI3DDetector:
    def __init__(self, rtsp_url, roi_coords=FIXED_ROI):
        """
        ROI tabanlı 3D nesne tespit sistemi
        
        Args:
            rtsp_url (str): RTSP kamera adresi
            roi_coords (list): ROI koordinatları [x1, y1, x2, y2]
        """
        self.rtsp_url = rtsp_url
        self.roi_coords = roi_coords
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # YOLOE model initialization
        print(f"🤖 YOLOE modeli yükleniyor... (Device: {self.device})")
        self.yolo_model = YOLOE("yoloe-v8l-seg.pt")
        if self.device == "cuda":
            self.yolo_model = self.yolo_model.cuda()
        self.yolo_model.set_classes(NAMES, self.yolo_model.get_text_pe(NAMES))
        
        # Depth estimator initialization
        print("🔍 Depth estimator yükleniyor...")
        self.depth_estimator = DepthEstimator(model_size='small', device=self.device)
        
        # 3D bbox estimator initialization
        print("📦 3D bbox estimator yükleniyor...")
        self.bbox_3d_estimator = BBox3DEstimator()
        
        print("✅ Tüm modeller başarıyla yüklendi!")

    def filter_detections_by_roi(self, detections, roi_bbox):
        """ROI içerisindeki tespitleri filtrele"""
        x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
        filtered_indices = []
        
        print(f"   🔍 ROI: [{x1_roi}, {y1_roi}, {x2_roi}, {y2_roi}]")
        
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            print(f"   📍 Detection {i}: center=({center_x:.0f}, {center_y:.0f}), bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            
            if x1_roi <= center_x <= x2_roi and y1_roi <= center_y <= y2_roi:
                filtered_indices.append(i)
                print(f"       ✅ ROI içinde!")
            else:
                print(f"       ❌ ROI dışında")
        
        if filtered_indices:
            return detections[filtered_indices]
        else:
            return sv.Detections.empty()

    def create_3d_bboxes(self, detections_2d, depth_map):
        """2D tespitlerden 3D bounding box'lar oluştur - gelişmiş derinlik hesaplama"""
        bboxes_3d = []
        
        # Depth map istatistikleri
        depth_min, depth_max = np.min(depth_map), np.max(depth_map)
        depth_range = depth_max - depth_min
        
        for i, bbox_2d in enumerate(detections_2d.xyxy):
            x1, y1, x2, y2 = map(int, bbox_2d)
            
            # Bbox merkezi
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # ROI içindeki derinlik analizi
            roi_depth = depth_map[y1:y2, x1:x2]
            
            # Çoklu derinlik ölçümü
            center_depth = depth_map[center_y, center_x]
            mean_depth = np.mean(roi_depth)
            min_depth = np.min(roi_depth)
            max_depth = np.max(roi_depth)
            depth_variance = np.std(roi_depth)
            
            # Gerçek mesafe hesaplama (kamera pozisyonuna göre)
            image_height, image_width = depth_map.shape
            
            # Y pozisyonuna göre mesafe tahmini (perspektif düzeltme)
            y_ratio = center_y / image_height
            
            # Kamera yüksekliği 2.5m, açı 15 derece varsayımı
            if y_ratio < 0.3:  # Üst kısım (uzak)
                base_distance = 3.0 + (0.3 - y_ratio) * 5.0  # 3-4.5m
            elif y_ratio < 0.7:  # Orta kısım
                base_distance = 1.5 + (0.7 - y_ratio) * 3.75  # 1.5-3m  
            else:  # Alt kısım (yakın)
                base_distance = 0.5 + (0.7 - y_ratio) * (-3.33)  # 0.5-1.5m
            
            # Derinlik map değeri ile ince ayar
            depth_normalized = (center_depth - depth_min) / (depth_range + 1e-6)
            depth_modifier = 0.7 + (depth_normalized * 0.6)  # 0.7-1.3 arası
            
            real_distance = base_distance * depth_modifier
            
            # Nesne boyutu hesaplama (pixel to meter)
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            
            # Mesafeye bağlı ölçek faktörü
            scale_factor = real_distance / 1000  # 1m'de 1000 pixel varsayımı
            
            real_width = pixel_width * scale_factor
            real_height = pixel_height * scale_factor
            real_depth_est = max(real_width, real_height) * 0.5  # Tahmini derinlik
            
            # 3D bbox tahmin et
            bbox_3d_data = self.bbox_3d_estimator.estimate_3d_box(
                bbox_2d=[x1, y1, x2, y2],
                depth_value=float(real_distance),
                class_name='object',
                object_id=i
            )
            
            if bbox_3d_data is not None:
                # Gelişmiş 3D data yapısı
                bbox_3d = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'depth': float(real_distance),
                    'depth_raw': float(center_depth),
                    'depth_stats': {
                        'mean': float(mean_depth),
                        'variance': float(depth_variance),
                        'range': float(max_depth - min_depth)
                    },
                    'real_size': {
                        'width_cm': real_width * 100,
                        'height_cm': real_height * 100,
                        'depth_cm': real_depth_est * 100
                    },
                    'center_3d': bbox_3d_data.get('center_3d', [0, 0, real_distance]),
                    'dimensions': [real_width, real_height, real_depth_est]
                }
                bboxes_3d.append(bbox_3d)
                
                # Debug çıktısı
                print(f"       📏 Nesne {i}: {real_distance:.2f}m, boyut: {real_width*100:.1f}x{real_height*100:.1f}cm")
        
        return bboxes_3d

    def draw_simple_3d_bbox(self, frame, bbox_3d):
        """Basit 3D bbox wireframe çizimi"""
        try:
            x1, y1, x2, y2 = bbox_3d['x1'], bbox_3d['y1'], bbox_3d['x2'], bbox_3d['y2']
            depth = bbox_3d['depth']
            
            # 3D effect için offset hesapla (depth'e göre)
            offset_x = int(min(30, max(10, depth * 5)))
            offset_y = int(min(20, max(8, depth * 3)))
            
            # Arka yüz koordinatları
            x1_back = x1 + offset_x
            y1_back = y1 - offset_y
            x2_back = x2 + offset_x
            y2_back = y2 - offset_y
            
            # Arka yüz çiz (mavi)
            cv2.rectangle(frame, (x1_back, y1_back), (x2_back, y2_back), (255, 0, 0), 2)
            
            # Bağlantı çizgileri (3D effect)
            cv2.line(frame, (x1, y1), (x1_back, y1_back), (0, 255, 255), 2)  # Sol üst
            cv2.line(frame, (x2, y1), (x2_back, y1_back), (0, 255, 255), 2)  # Sağ üst
            cv2.line(frame, (x1, y2), (x1_back, y2_back), (0, 255, 255), 2)  # Sol alt
            cv2.line(frame, (x2, y2), (x2_back, y2_back), (0, 255, 255), 2)  # Sağ alt
            
            # Merkez noktası ve derinlik info
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # 3D bilgi
            cv2.putText(frame, f"3D Box", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                       
        except Exception as e:
            print(f"3D çizim hatası: {e}")
            pass

    def draw_roi_and_info(self, frame, roi_detections_3d, all_detections_count):
        """ROI ve bilgi çizimi"""
        # ROI çiz (kırmızı dikdörtgen)
        cv2.rectangle(frame, (self.roi_coords[0], self.roi_coords[1]), 
                     (self.roi_coords[2], self.roi_coords[3]), (0, 0, 255), 3)
        
        # ROI bilgisi
        roi_width = self.roi_coords[2] - self.roi_coords[0]
        roi_height = self.roi_coords[3] - self.roi_coords[1]
        cv2.putText(frame, f"ROI: {roi_width}x{roi_height}px", 
                   (self.roi_coords[0], self.roi_coords[1]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Frame boyutu ve tespit sayısı
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Frame: {width}x{height}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Detections: {all_detections_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"ROI 3D Detections: {len(roi_detections_3d)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

    def run_detection(self):
        """Ana tespit döngüsü"""
        # Kamera bağlantısı
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # RTSP performans ayarları
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # RTSP timeout ayarları
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 saniye timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 saniye read timeout
        
        if not cap.isOpened():
            print("❌ Kamera açılamadı! RTSP adresini kontrol edin.")
            print("🔄 Webcam ile test etmek için test_webcam_3d.py kullanın")
            return
        
        print("📹 Kamera bağlantısı başarılı! 'q' tuşuna basarak çıkabilirsiniz.")
        print("🎯 3D ROI Detection başlatılıyor...")
        
        frame_count = 0
        skip_frames = 2  # Her 2 frame'de bir AI işlemi (daha az skip)
        
        # Son tespit sonuçlarını sakla (smoothing için)
        last_detections_2d = sv.Detections.empty()
        last_roi_detections_3d = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Kare alınamadı! Yeniden bağlanmaya çalışılıyor...")
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            frame_count += 1
            
            # Her skip_frames'de bir AI işlemi yap
            if frame_count % skip_frames == 0:
                try:
                    print(f"🔄 Frame {frame_count} işleniyor...")
                    
                    # 1. YOLOE ile 2D tespit
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    results = self.yolo_model.predict(pil_image, conf=0.15, verbose=False)
                    detections_2d = sv.Detections.from_ultralytics(results[0])
                    
                    print(f"   📊 2D Detections: {len(detections_2d)}")
                    
                    # 2. Depth estimation
                    depth_map = self.depth_estimator.estimate_depth(rgb_frame)
                    print(f"   🔍 Depth map shape: {depth_map.shape}")
                    
                    # 3. ROI filtreleme (2D)
                    roi_detections_2d = self.filter_detections_by_roi(detections_2d, self.roi_coords)
                    print(f"   🎯 ROI 2D Detections: {len(roi_detections_2d)}")
                    
                    # 4. 3D bbox oluşturma (ROI içindekiler için)
                    roi_detections_3d = self.create_3d_bboxes(roi_detections_2d, depth_map)
                    print(f"   📦 ROI 3D Detections: {len(roi_detections_3d)}")
                    
                    # Sonuçları güncelle
                    last_detections_2d = detections_2d
                    last_roi_detections_3d = roi_detections_3d
                    
                except Exception as e:
                    print(f"⚠️  İşleme hatası: {e}")
                    # Önceki sonuçları kullan
                    detections_2d = last_detections_2d
                    roi_detections_3d = last_roi_detections_3d
            else:
                # Önceki sonuçları kullan (smooth transition)
                detections_2d = last_detections_2d
                roi_detections_3d = last_roi_detections_3d
            
            # 5. Görselleştirme (her frame'de)
            annotated_frame = frame.copy()
            
            # Tüm 2D tespitleri gri renkte çiz
            if len(detections_2d) > 0:
                for bbox in detections_2d.xyxy:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            
            # ROI içindeki 3D tespitleri çiz
            for i, bbox_3d in enumerate(roi_detections_3d):
                # 2D bbox (yeşil)
                cv2.rectangle(annotated_frame, (bbox_3d['x1'], bbox_3d['y1']), 
                            (bbox_3d['x2'], bbox_3d['y2']), (0, 255, 0), 3)
                
                # Gelişmiş bilgi gösterimi
                depth = bbox_3d['depth']
                width_cm = bbox_3d['real_size']['width_cm']
                height_cm = bbox_3d['real_size']['height_cm']
                depth_variance = bbox_3d['depth_stats']['variance']
                
                # Mesafe bilgisi
                cv2.putText(annotated_frame, f"Mesafe: {depth:.2f}m", 
                           (bbox_3d['x1'], bbox_3d['y1']-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Boyut bilgisi
                cv2.putText(annotated_frame, f"Boyut: {width_cm:.1f}x{height_cm:.1f}cm", 
                           (bbox_3d['x1'], bbox_3d['y1']-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Derinlik kalitesi
                quality = "Yüksek" if depth_variance < 0.02 else "Düşük"
                cv2.putText(annotated_frame, f"Kalite: {quality}", 
                           (bbox_3d['x1'], bbox_3d['y1']+bbox_3d['y2']-bbox_3d['y1']+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # 3D wireframe çiz (basit implementasyon)
                self.draw_simple_3d_bbox(annotated_frame, bbox_3d)
            
            # ROI ve bilgi çizimi
            annotated_frame = self.draw_roi_and_info(annotated_frame, roi_detections_3d, len(detections_2d))
            
            # Sonucu göster
            cv2.imshow('3D ROI Metal Detection', annotated_frame)
            
            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Ana fonksiyon"""
    # RTSP adresi
    rtsp_url = "rtsp://admin:HeysemAI246@192.168.150.59"
    
    # 3D detector oluştur ve çalıştır
    detector = ROI3DDetector(rtsp_url)
    detector.run_detection()

if __name__ == "__main__":
    main()
