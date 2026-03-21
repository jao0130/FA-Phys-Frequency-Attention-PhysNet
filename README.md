# FA-Phys: Frequency Attention PhysNet

FA-Phys is a lightweight deep learning framework for **remote physiological signal estimation (rPPG)** from facial videos.  
The model introduces a **Frequency Attention Module** that explicitly focuses on physiologically meaningful heart-rate frequency bands, enabling robust and accurate prediction under motion, illumination, and background variations.
# Workfolws
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/workflow_architecture.png)
# Model Architecture
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/image.png)
# FAMask Module Architecture
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/fam_architecture.png)
---

## 🔍 Motivation
Remote photoplethysmography aims to estimate physiological signals such as heart rate (HR) and blood oxygen saturation (SpO₂) without physical contact.  
However, real-world scenarios suffer from motion noise, illumination changes, and background interference.

FA-Phys addresses these challenges by:
- Incorporating **frequency-domain priors** into deep learning
- Enhancing temporal modeling with minimal computational overhead
- Achieving strong performance with significantly fewer parameters

![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/att.png)

---

## 🚀 Key Contributions
- **Frequency Attention Module**
  - Focuses on physiologically meaningful frequency bands (0.7–3.0 Hz)
  - Suppresses motion-induced and irrelevant frequency components

- **Channel Attention**
  - Dynamically emphasizes critical physiological feature channels

- **Temporal Shift Module (TSM)**
  - Improves long-range temporal dependency modeling
  - No additional 3D convolution computation cost

- **Multi-task Learning**
  - Simultaneous estimation of HR, SpO₂, and blood pressure

---

## 📊 Performance Highlights
| Dataset        | Task | MAE |
|---------------|------|-----|
| PURE          | HR   | 0.22 bpm |
| UBFC-rPPG     | HR   | 0.66 bpm |
| Multiple Datasets | HR | +62.07% improvement over TS-CAN |

- Achieves comparable or superior accuracy with **92% fewer parameters than PhysFormer**

---

## 🛠️ Tech Stack
- PyTorch
- OpenCV, NumPy
- FFT-based Frequency Analysis
- CNN / Attention-based Architectures

---

## 📁 Supported Datasets
- PURE
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/PURE.png)
- UBFC-rPPG
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/UBFC.png)
- SUMS
![image](https://github.com/jao0130/FA-Phys-Frequency-Attention-PhysNet/blob/main/image/SUMS.png)

---

## 📌 Applications
- Smart Healthcare (remote patient monitoring)
- Intelligent Transportation (driver fatigue detection)
- Vision-based physiological sensing

---

## 📜 License
This project is intended for research and academic use.

---

## 👤 Author
**Tzu-Yao Chang**  
M.S. in Electrical Engineering, National Taiwan Ocean University  
