# FPI ROI Metal Detection - 3D Enhanced System

Advanced ROI-based metal detection system with **3D visualization capabilities** for IP cameras and webcams.

## 🚀 Features

### Core Detection
- **YOLOE (YOLO with Embeddings)**: Text-guided object detection
- **ROI-based filtering**: Focused detection within defined regions
- **Real-time processing**: Optimized for live camera feeds
- **Multi-camera support**: RTSP IP cameras and USB webcams

### 3D Capabilities ✨
- **Monocular depth estimation**: Using Depth Anything v2 model
- **3D bounding boxes**: Real-time 3D visualization
- **Wireframe rendering**: Perspective-based 3D wireframes
- **Depth information**: Distance measurements for detected objects

### Performance Optimizations
- **CUDA acceleration**: GPU-accelerated inference
- **Smooth transitions**: Optimized frame processing (2-frame skipping)
- **Result caching**: Seamless visual experience
- **Background processing**: Non-blocking AI operations

## 📁 Project Structure

```
FPI-ROI-Metal-Detection/
├── src/
│   ├── main.py              # Static image analysis
│   ├── cam_3d.py           # 3D RTSP camera detection
│   ├── test_webcam_3d.py   # 3D webcam testing
│   ├── depth_model.py      # Depth estimation module
│   ├── bbox3d_utils.py     # 3D bbox utilities
│   └── load_camera_params.py # Camera calibration
├── models/                  # AI model files
├── data/                   # Sample images
├── output/                 # Detection results
└── requirements.txt        # Dependencies
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/omerkurueks/FPI-ROI-Metal-Detection.git
cd FPI-ROI-Metal-Detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download model files:**
   - Place `yoloe-v8l-seg.pt` in `models/` directory
   - Place `mobileclip_blt.pt` in project root

## 🎯 Usage

### 3D RTSP Camera Detection
```bash
python src/cam_3d.py
```
- Connects to IP camera via RTSP
- Real-time 3D object detection
- ROI filtering and visualization

### 3D Webcam Testing
```bash
python src/test_webcam_3d.py
```
- Uses local webcam for testing
- 3D wireframe visualization
- Performance monitoring

### Static Image Analysis
```bash
python src/main.py
```
- Analyze single images
- ROI-based detection
- Save results to output/

## 📊 System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB recommended for smooth operation)
- **Python**: 3.8+
- **OS**: Windows/Linux/macOS

## 🔍 Detection Process

1. **2D Object Detection**: YOLOE identifies objects in frame
2. **ROI Filtering**: Filter detections within defined region
3. **Depth Estimation**: Calculate depth map using Depth Anything v2
4. **3D Bbox Creation**: Generate 3D bounding boxes with depth info
5. **Visualization**: Render wireframes and depth information

## 🎨 Visualization Features

- **2D Bboxes**: Green rectangles for standard detections
- **3D Wireframes**: Blue perspective frames with yellow connections
- **Depth Info**: Distance measurements in meters
- **ROI Boundaries**: Red rectangle showing detection zone
- **Smooth Transitions**: Optimized frame processing for fluid visualization

## ⚙️ Configuration

### Camera Settings (cam_3d.py)
```python
rtsp_url = "rtsp://admin:password@192.168.1.100"
roi_coords = [200, 150, 600, 450]  # [x1, y1, x2, y2]
```

### Performance Tuning
```python
skip_frames = 2  # Process every 2nd frame
confidence = 0.2  # Detection confidence threshold
```

## 🔬 Technical Details

### Models Used
- **YOLOE**: YOLOv8-based with embedding support
- **Depth Anything v2**: Transformer-based depth estimation
- **MobileClip**: Efficient CLIP model for text-image matching

### Processing Pipeline
1. Frame capture (RTSP/Webcam)
2. YOLOE inference on GPU
3. Depth map generation
4. ROI intersection calculation
5. 3D bbox estimation
6. Wireframe rendering
7. Display with smooth transitions

## 🐛 Troubleshooting

### Common Issues
- **CUDA not available**: Install CUDA-compatible PyTorch
- **Model files missing**: Download required .pt files
- **RTSP connection failed**: Check camera IP and credentials
- **Low performance**: Reduce frame resolution or increase skip_frames

### Performance Tips
- Use GPU acceleration when available
- Adjust ROI size for optimal detection
- Monitor system resources during operation
- Close unnecessary applications for better performance

## 📈 Future Enhancements

- [ ] Multi-object tracking in 3D space
- [ ] Advanced depth filtering algorithms
- [ ] Web interface for remote monitoring
- [ ] Alert system integration
- [ ] Historical data logging
- [ ] Mobile app companion

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO**: Ultralytics YOLOv8 framework
- **Depth Anything v2**: Advanced monocular depth estimation
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library

---

**Created by**: Ömer Kuru  
**Project**: FPI ROI Metal Detection System  
**Version**: 2.0 (3D Enhanced)
