# Auto-Attack-and-Digital-Defense

## Closed-Loop XAI-Guided Laser Attack + Digital Color-Filter Defenses

This repository implements a **general closed-loop physical attack framework** for vision-based monitoring systems. It performs **autonomous, XAI-guided laser perturbations** and includes a full suite of **digital color-filter defenses**. The system is device-agnostic: adapting to a new device only requires replacing classifier weight files.

---
## 1. Repository Structure
```
Auto-Attack-and-Digital-Defense/
│
├── Device-A/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── Device-B/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── Device-C/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── README.md
├── servo_api_server.py
└── servo_gamepad_tamed.py
```
### Important
PC folder must contain:
- `model_def.py`
- `panel_cls_full.pt`
- `panel_cls_surrogate.pt`
- `xai_closedloop_pc.py`

Raspberry Pi only needs:
- `servo_api_server.py`
- *(optional)* `servo_gamepad_tamed.py`

---
## 2. System Overview
A real-time closed-loop XAI-guided physical laser attack:
1. Camera captures target panel.
2. PC computes **Grad-CAM** on surrogate model to select sensitive pixel.
3. PC tracks real laser position.
4. PC sends micro-step commands to Raspberry Pi.
5. Pi controls yaw/pitch servos and red/green lasers.

Digital defenses are applied to the ROI before running the classifier.

---
## 3. PC Requirements
```
pip install torch torchvision opencv-python numpy
python3 xai_closedloop_pc.py
```
---
## 4. Raspberry Pi Setup
```
sudo apt install pigpio
pip install pigpio pygame
sudo pigpiod
python3 servo_api_server.py
```
Optional:
```
python3 servo_gamepad_tamed.py
```

---
## 5. Closed-Loop Attack Operation
**ROI Selection**
- Draw ROI, press `s` to lock, `q` to exit.

**Controls**
| Key | Action |
|-----|--------|
| 1 | Auto (Grad-CAM) target |
| 2 | Manual target |
| r | Reset lock |
| f | Cycle defenses |
| q | Quit |

Overlays show predictions, Grad-CAM pixel, laser tracking, active filter, and probabilities.

---
## 6. Digital Color-Filter Defenses
A full suite of digital defenses used for evaluation.

### 6.1 Baseline
**none** — raw ROI.

### 6.2 RGB Channel Removal
- `remove_green`
- `remove_red`

Zeroes channel to erase laser color.

### 6.3 Strong Channel Suppression
- `remove_green_strong`
- `remove_red_strong`

Zero channel + HSV mask + aggressive brightness reduction.

### 6.4 HSV Blocking
- `hsv_block_green`
- `hsv_block_red`

Dims laser-colored regions while preserving image content.

### 6.5 Laser Inpainting (Best Defense)
- `laser_inpaint_green`
- `laser_inpaint_green_strong`
- `laser_inpaint_red`
- `laser_inpaint_red_strong`

Detect laser mask → dilate → OpenCV inpainting.

### 6.6 Highlight Clipping
- `clip_highlights`

Color-agnostic removal of bright saturated pixels.

### 6.7 Grayscale Defenses
- `gray`
- `gray_blur`

Removes color cues; blur removes small artifacts.

---
## 7. Main vs Surrogate Model
| Component | Purpose |
|----------|---------|
| Main model (`panel_cls_full.pt`) | Primary classifier |
| Surrogate model (`panel_cls_surrogate.pt`) | Used for Grad-CAM + transferability testing |

Digital defenses apply to both models.

---
## 8. Supported Devices
Extensible to any device.
Only classifier weight files need updating.

---
## 9. Summary
This repository provides:
- Closed-loop XAI-guided physical attack framework
- Real-time servo + laser control
- 15+ digital defenses
- Attack and defense transferability evaluation
- Unified, reproducible pipeline for physical adversarial ML research

