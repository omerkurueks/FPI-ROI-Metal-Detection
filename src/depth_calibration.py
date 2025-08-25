"""
Derinlik kalibrasyonu için yardımcı fonksiyonlar
"""
import numpy as np
import cv2

class DepthCalibrator:
    def __init__(self, camera_height=2.0, camera_angle=0):
        """
        Args:
            camera_height (float): Kamera yüksekliği (metre)
            camera_angle (float): Kamera açısı (derece)
        """
        self.camera_height = camera_height
        self.camera_angle = np.radians(camera_angle)
        
        # Kamera kalibrasyonu - gerçek ölçümlerle ayarla
        self.focal_length = 800  # pixels (kameraya göre değişir)
        self.real_world_width = 3.0  # metre (görüş alanı genişliği)
        
    def normalize_depth_to_real_distance(self, depth_map, bbox_center):
        """
        Normalize edilmiş derinlik değerini gerçek mesafeye çevir
        """
        center_x, center_y = bbox_center
        
        # Normalize depth değeri (0-1)
        norm_depth = depth_map[center_y, center_x]
        
        # Kamera parametreleri
        image_height, image_width = depth_map.shape
        
        # Perspective transformation
        # Y ekseni: kameradan zemine doğru
        y_ratio = center_y / image_height
        
        # Basit trigonometri ile gerçek mesafe
        # Bu formül kamera açısına ve yüksekliğine göre ayarlanmalı
        if y_ratio > 0.5:  # Alt yarı (daha yakın)
            base_distance = 0.5 + (y_ratio - 0.5) * 2.0  # 0.5-2.5m
        else:  # Üst yarı (daha uzak)
            base_distance = 2.5 + (0.5 - y_ratio) * 3.0  # 2.5-5.5m
            
        # Depth map değeri ile modifikasyon
        depth_modifier = 0.8 + (norm_depth * 0.4)  # 0.8-1.2 arası
        
        real_distance = base_distance * depth_modifier
        
        return real_distance
    
    def calculate_object_size(self, bbox_2d, real_distance):
        """
        Gerçek mesafeye göre nesne boyutunu hesapla
        """
        x1, y1, x2, y2 = bbox_2d
        
        # Pixel boyutları
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        # Gerçek dünya boyutları (basit oran)
        # Bu kamera kalibrasyonuna göre ayarlanmalı
        meters_per_pixel = real_distance / 1000  # Kaba tahmin
        
        real_width = pixel_width * meters_per_pixel
        real_height = pixel_height * meters_per_pixel
        
        return {
            'width_cm': real_width * 100,
            'height_cm': real_height * 100,
            'area_cm2': real_width * real_height * 10000
        }
    
    def detect_depth_variance(self, bbox_2d, depth_map):
        """
        Nesne içindeki derinlik varyansını analiz et
        """
        x1, y1, x2, y2 = map(int, bbox_2d)
        
        # ROI içindeki derinlik değerleri
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # İstatistikler
        stats = {
            'mean_depth': np.mean(roi_depth),
            'std_depth': np.std(roi_depth),
            'min_depth': np.min(roi_depth),
            'max_depth': np.max(roi_depth),
            'depth_range': np.max(roi_depth) - np.min(roi_depth)
        }
        
        # Nesne tipi tahmini
        if stats['std_depth'] < 0.01:
            object_type = "flat_surface"  # Düz yüzey
        elif stats['depth_range'] > 0.1:
            object_type = "3d_object"     # 3D nesne
        else:
            object_type = "curved_surface"  # Kavisli yüzey
            
        stats['estimated_type'] = object_type
        
        return stats

class StereoDepthCalculator:
    """
    Gelecekte iki kamera kullanmak için
    """
    def __init__(self, baseline=0.1, focal_length=800):
        self.baseline = baseline  # Kameralar arası mesafe (metre)
        self.focal_length = focal_length
        
    def calculate_stereo_depth(self, disparity):
        """
        Stereo vision ile gerçek derinlik hesapla
        """
        # Z = (f * B) / d
        # Z: derinlik, f: focal length, B: baseline, d: disparity
        depth = (self.focal_length * self.baseline) / (disparity + 1e-6)
        return depth

# Kullanım örneği
def enhanced_depth_processing():
    """
    Gelişmiş derinlik işleme örneği
    """
    calibrator = DepthCalibrator(camera_height=2.5, camera_angle=15)
    
    # Örnek kullanım
    example_depth_map = np.random.rand(480, 640)  # Simüle depth map
    bbox_center = (320, 240)
    bbox_2d = [300, 220, 340, 260]
    
    # Gerçek mesafe hesapla
    real_distance = calibrator.normalize_depth_to_real_distance(
        example_depth_map, bbox_center
    )
    
    # Nesne boyutları
    object_size = calibrator.calculate_object_size(bbox_2d, real_distance)
    
    # Derinlik analizi
    depth_stats = calibrator.detect_depth_variance(bbox_2d, example_depth_map)
    
    return {
        'real_distance_m': real_distance,
        'object_size': object_size,
        'depth_analysis': depth_stats
    }

if __name__ == "__main__":
    result = enhanced_depth_processing()
    print("Enhanced Depth Analysis:", result)
