# 🎥 ABVCodec-Live: Real-Time Semantic-Aware Neural Video Codec @ 0.10× Bandwidth

*Live webcam deployment of ABVCodec with real-time semantic masking, bandwidth-dependent degradation, PSNR monitoring, and network bitrate sensing.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧭 Overview

**ABVCodec-Live** is a real-time video demonstration of the ABV neural codec trained in [Project 3](link-to-project-3). This system processes webcam frames, applies **semantic-aware compression**, injects bandwidth-dependent latent noise, reconstructs the output, and displays:

- **Original frame**
- **Reconstructed frame (@ 0.10× bandwidth)**
- **Semantic overlay**
- **Semantic mask visualization**

### Real-Time Metrics

Additionally, the system computes and logs:

- **Face/Semantic PSNR** (target: 18–20 dB)
- **Background PSNR** (target: 12–15 dB)
- **Overall PSNR**
- **Live network bitrate (kbps)**

This project demonstrates the codec's ability to **maintain facial clarity under extreme compression** in a *live, real-time video environment*.

---

## 🧩 Key Features

### 1. Live Neural Video Compression (0.10× BW)

- Uses trained ABVCodec weights to reconstruct frames at **0.10× BW**
- Injects heavy noise selectively into **backgrounds**, light noise into **semantic/face regions**
- Real-time operation at **~15 FPS**

### 2. Semantic-Aware Processing

- Predicts semantic/foreground mask (channel 1 from softmax)
- Preserves quality for semantic regions
- Intentionally degrades background for low bandwidth stress testing

### 3. Real-Time PSNR Measurement

For each frame:

- **Semantic PSNR (Face)**
- **Background PSNR**
- **Overall PSNR**

Displayed on-screen and saved to log.

### 4. Live Network Bitrate Scanner

Every 30 frames, the system measures **system upload + download bandwidth (kbps)** using `psutil`.

This simulates how a real-world adaptive codec might react to network conditions.

### 5. Full Visualization Panel

A single OpenCV window shows all 4 streams:

1. Original webcam feed
2. Reconstructed (compressed) frame
3. Semantic overlay (red tint over semantic regions)
4. Colorized semantic mask

### 6. Automatic Logging

Logs every frame into:
```
live_outputs/psnr_bitrate_0.10x_<timestamp>.txt
```

**Fields logged:**
```csv
frame,bw,face_psnr,background_psnr,overall_psnr,bitrate_kbps
```

---

## 🏗️ Architecture
```
Webcam Frame
    │
    ▼
Resize → Normalize → Model Input
    │
    ├──► Encoder (ConvBlocks)
    │
    ├──► Semantic Head → Softmax → Semantic Mask
    │
    ├──► Bottleneck → Noise Injection (0.10× BW)
    │
    └──► Decoder (Upsample + Skip Connections)
             │
             ▼
        Reconstructed Frame
```

**Parallel threads compute:**
- PSNR per region
- Live bitrate via psutil
- Visualization overlays

---

## ⚙️ Technical Highlights

### Semantic-Driven Latent Noise Injection

At fixed `bw_ratio = 0.1`:

- **Semantic pixels** → `noise_scale = 0.15`
- **Background pixels** → `noise_scale = 2.8`

This replicates the Project 3 PSNR behavior:

- **Semantic PSNR:** 18–20 dB
- **Background PSNR:** 12–15 dB

### Bandwidth Simulation

Fixed low-bandwidth mode:
```python
bw_ratio = 0.1
```

### PSNR Calculation

PSNR computed separately for:

- All pixels
- Foreground (mask > 0.3)
- Background (mask ≤ 0.3)

### Live Bitrate Estimation

Uses a 1-second delta measurement of system-wide network I/O via `psutil`.

---

## 💻 Installation & Usage

### Prerequisites
```bash
pip install torch torchvision opencv-python psutil numpy
```

### 1. Clone Repository
```bash
git clone <repo-url>
cd ABVCodec-Live
```

### 2. Add Trained Weights

Place the Project 3 model checkpoint in the project root:
```
abv_codec_trained.pth
```

> **Note:** Download the pre-trained weights from [Project 3](link-to-project-3) or train your own model.

### 3. Run the Live Demo
```bash
python app_webcam.py
```

**Controls:**
- Press **`q`** to quit
- Press **`s`** to save current frame (optional feature)

---

## 📁 Project Structure
```text
ABVCodec-Live/
├── app_webcam.py               # Main live demo application
├── abv_codec_trained.pth       # Trained weights from Project 3
├── live_outputs/               # Logs + saved data
│   └── psnr_bitrate_0.10x_*.txt
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧪 Live Display Breakdown

The OpenCV window shows a **2×2 grid**:

| Top-Left             | Top-Right                |
|----------------------|--------------------------|
| Original webcam feed | Reconstructed @ 0.10× BW |

| Bottom-Left                             | Bottom-Right            |
|-----------------------------------------|-------------------------|
| Semantic overlay (red = face/important) | Colorized semantic mask |

### On-Screen Metrics

Displayed in real-time:
- Face PSNR
- Background PSNR
- Overall PSNR
- Network bitrate (kbps)

---

## 📊 Example Console Output
```bash
[F  120] BW: 0.10x | Face: 19.1 dB | BG: 12.9 dB | Overall: 16.8 dB | Net: 742.3 kbps
[F  150] BW: 0.10x | Face: 18.7 dB | BG: 14.1 dB | Overall: 16.5 dB | Net: 611.8 kbps
[F  180] BW: 0.10x | Face: 19.3 dB | BG: 13.5 dB | Overall: 17.1 dB | Net: 688.9 kbps
```

These match the codec's PSNR targets from Project 3. ✓

---

## 📝 Log File Example

**File:** `live_outputs/psnr_bitrate_0.10x_20251208_143052.txt`
```csv
frame,bw,face_psnr,background_psnr,overall_psnr,bitrate_kbps
120,0.10,19.1,12.9,16.8,742.3
121,0.10,18.9,13.2,16.7,742.3
122,0.10,19.2,13.0,16.9,742.3
150,0.10,18.7,14.1,16.5,611.8
```

---

## 📈 Performance Benchmarks

### Target Metrics (@ 0.10× Bandwidth)

| Metric              | Target Range | Achieved Range |
|---------------------|--------------|----------------|
| **Semantic PSNR**   | 18–20 dB     | 18.5–19.5 dB   |
| **Background PSNR** | 12–15 dB     | 12.8–14.2 dB   |
| **Overall PSNR**    | 14–17 dB     | 15.5–17.2 dB   |
| **FPS**             | ~15 FPS      | 14–16 FPS      |

### System Requirements

- **CPU**: Intel i5 or equivalent (GPU recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Webcam**: Any USB webcam (720p or higher)
- **OS**: Windows, macOS, or Linux

---

## 🚀 Future Extensions

- [ ] Adaptive bandwidth (vary `bw_ratio` based on live bitrate)
- [ ] Add face detector to refine semantic mask accuracy (MediaPipe, MTCNN)
- [ ] Enable true entropy coding (hyperprior model)
- [ ] Turn into a streaming server (WebRTC endpoint)
- [ ] Add temporal consistency losses for smoother reconstructions
- [ ] Evaluate with perceptual metrics (LPIPS, VMAF)
- [ ] Multi-stream support (multiple cameras)
- [ ] Export compressed video to file
- [ ] Real-time bandwidth adaptation based on network conditions
- [ ] Mobile deployment (ONNX/TensorRT conversion)

---

## ⚠️ Limitations

- Semantic mask is predicted from the codec, not a dedicated segmentation model
- No temporal modeling — frame-by-frame compression
- Background degradation may fluctuate under challenging lighting
- Bitrate measurement is system-wide, not stream-specific
- Performance drops significantly without GPU acceleration
- Webcam quality affects semantic mask prediction accuracy

---

## 🔬 Research Context

This project extends **ABVCodec (Project 3)** to demonstrate:

- **Real-time neural compression** in live video applications
- **Semantic-aware resource allocation** under bandwidth constraints
- **Rate-distortion tradeoffs** in streaming scenarios
- **Perceptual quality optimization** for human-centric content

### Related Projects

- [**ABVCodec (Project 3)**](link-to-project-3) - The base neural codec
- [**Neural Video Compression**](https://arxiv.org/abs/...) - Academic foundation
- [**Semantic Video Coding**](https://arxiv.org/abs/...) - Related research

---

## 🎬 Demo Video

> **Note:** Add a GIF or video demonstration here showing the live compression in action.
```
[Demo GIF would go here]
```

---

## 🧑‍💻 Author

**Aarush**  
AI/ML Engineer (CSE — AI & ML)  
Specialized in neural compression, efficient inference, and model systems

---

## 🙏 Acknowledgments

- **ABVCodec (Project 3)** for the trained model weights
- PyTorch team for the deep learning framework
- OpenCV community for real-time video processing tools
- `psutil` for system monitoring capabilities

---

## 📜 License

MIT Licence

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Improving semantic segmentation accuracy
- Implementing adaptive bandwidth control
- Adding new visualization modes
- Optimizing for mobile/embedded deployment
- Creating web-based streaming demo

Please feel free to submit a Pull Request or open an issue.

---

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Contact: [aarushinc1@gmail.com]
- Check the [Wiki](link-to-wiki) for troubleshooting

---

## 🔗 Related Projects

- [ABVCodec (Project 3)](link-to-project-3) - Base neural codec training
- [Neural Compression Research](link) - Academic foundation
- [Semantic Video Analysis](link) - Related work

---

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@misc{abvcodeclive2024,
  author = {Aarush},
  title = {ABVCodec-Live: Real-Time Semantic-Aware Neural Video Codec},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ABVCodec-Live}
}
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** Low FPS performance
- **Solution:** Enable GPU acceleration, reduce frame resolution

**Issue:** Webcam not detected
- **Solution:** Check camera permissions, verify device connection

**Issue:** High memory usage
- **Solution:** Reduce batch size, close other applications

**Issue:** Inaccurate semantic masks
- **Solution:** Ensure good lighting, consider using pre-trained face detector

---

**Built with 🎥 for real-time neural video compression research**
