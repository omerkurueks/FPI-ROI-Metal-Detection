import cv2
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
from ultralytics import YOLOE
import torch
from datetime import datetime
import os

# Sabit ROI koordinatları (main.py ile aynı olmalı)
FIXED_ROI = [235, 1010, 780, 1241]  # [x1, y1, x2, y2]
NAMES = [
    "metal bar", "steel bar", "iron bar", "rectangular prism", "metal block",
    "steel beam", "iron beam", "metal piece", "industrial part", "steel product",
    "hand", "pen", "pencil", "notebook", "book", "cup", "mug", "glass", 
    "bottle", "water bottle", "beverage bottle"
]

def filter_detections_by_roi(detections, roi_bbox):
    x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
    filtered_indices = []
    for i, bbox in enumerate(detections.xyxy):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if x1_roi <= center_x <= x2_roi and y1_roi <= center_y <= y2_roi:
            filtered_indices.append(i)
    if filtered_indices:
        return detections[filtered_indices]
    else:
        return sv.Detections.empty()

def draw_roi_on_image(image, roi_bbox, color=(255, 0, 0), thickness=4):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = roi_bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    roi_width = x2 - x1
    roi_height = y2 - y1
    text = f"ROI: {roi_width}x{roi_height}px"
    draw.text((x1, y1-25), text, fill=color)
    return image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLOE("yoloe-v8l-seg.pt")
    if device == "cuda":
        model = model.cuda()
    model.set_classes(NAMES, model.get_text_pe(NAMES))

    # RTSP adresini kullanıcıdan al
    rtsp_url = "rtsp://admin:HeysemAI246@169.254.4.56"
    cap = cv2.VideoCapture(rtsp_url)
    
    # RTSP performans ayarları
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer size küçük tut
    cap.set(cv2.CAP_PROP_FPS, 30)        # FPS sınırla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Çözünürlük ayarla
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Kamera açılamadı! RTSP adresini kontrol edin.")
        return
    
    print("Kamera bağlantısı başarılı! 'q' tuşuna basarak çıkabilirsiniz.")
    frame_count = 0
    skip_frames = 2  # Her 2 frame'de bir AI işlemi yap

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı! Bağlantı koptu veya kamera erişilemiyor.")
            # Yeniden bağlanmayı dene
            cap.release()
            print("Yeniden bağlanmaya çalışılıyor...")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_count += 1
        # Her skip_frames'de bir AI işlemi yap (performans için)
        if frame_count % skip_frames == 0:
            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            results = model.predict(pil_image, conf=0.1, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            roi_detections = filter_detections_by_roi(detections, FIXED_ROI)
            annotated_image = pil_image.copy()
            annotated_image = draw_roi_on_image(annotated_image, FIXED_ROI, color=(255, 0, 0), thickness=4)
            if len(detections) > 0:
                annotated_image = sv.BoxAnnotator(color=sv.Color.from_hex("#808080")).annotate(
                    scene=annotated_image, detections=detections)
            if len(roi_detections) > 0:
                annotated_image = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=4).annotate(
                    scene=annotated_image, detections=roi_detections)
                annotated_image = sv.LabelAnnotator(color=sv.Color.from_hex("#00FF00")).annotate(
                    scene=annotated_image, detections=roi_detections)
            # PIL -> OpenCV
            annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        else:
            # Sadece ROI'yi göster (AI işlemi yapmadan)
            annotated_frame = frame.copy()
        
        # Her frame'de ROI'yi çiz (görünür olması için)
        cv2.rectangle(annotated_frame, (FIXED_ROI[0], FIXED_ROI[1]), 
                     (FIXED_ROI[2], FIXED_ROI[3]), (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"ROI: {FIXED_ROI[2]-FIXED_ROI[0]}x{FIXED_ROI[3]-FIXED_ROI[1]}px", 
                   (FIXED_ROI[0], FIXED_ROI[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Frame boyutunu göster (debug için)
        height, width = frame.shape[:2]
        cv2.putText(annotated_frame, f"Frame: {width}x{height}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('ROI Metal Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
