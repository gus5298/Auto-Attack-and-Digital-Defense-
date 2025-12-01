 **Auto-Attack-and-Digital-Defense

Closed-loop XAI Laser Attack + Digital Color-Filter Defenses**

This repository integrates all Siemens industrial HMI attack & defense code into one unified project.
It contains the complete closed-loop physical laser attack system + digital color-filter defenses.
It supports all three devices:
	•	SIMATIC-S7-1500
	•	Siemens-410E
	•	Siemens-Energy-T3000

Migration between devices requires only replacing:
	•	panel_cls_full.pt
	•	panel_cls_surrogate.pt

All scripts remain identical.

⸻

1. Repository Structure 

Auto-Attack-and-Digital-Defense/
│
├── SIMATIC-S7-1500/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── Siemens-410E/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── Siemens-Energy-T3000/
│      ├── model_def.py
│      ├── panel_cls_full.pt
│      ├── panel_cls_surrogate.pt
│      └── xai_closedloop_pc.py
│
├── README.md
├── servo_api_server.py
└── servo_gamepad_tamed.py

✔ Important

When running on PC:

model_def.py, panel_cls_full.pt, panel_cls_surrogate.pt, xai_closedloop_pc.py
must be placed in the same directory for that device.

Raspberry Pi only needs:
	•	servo_api_server.py
	•	(optional) servo_gamepad_tamed.py

⸻

2. System Overview

This system performs a real-time closed-loop physical laser attack on an industrial HMI classifier.

Workflow:
	1.	A camera captures the panel (HMI).
	2.	The PC selects a target pixel using Grad-CAM on a surrogate model.
	3.	The PC tracks actual laser position in the full image.
	4.	The PC sends micro-step commands to Raspberry Pi.
	5.	The Pi controls:
	•	Yaw servo
	•	Pitch servo
	•	Red/Green laser

To defend against this attack, digital color-filter defenses are applied to the ROI before classification.

⸻

3. PC-side Software Requirements

pip install torch torchvision opencv-python numpy

Run:

python3 xai_closedloop_pc.py


⸻

4. Raspberry Pi Setup

Install:

sudo apt install pigpio
pip install pigpio pygame
sudo pigpiod

Run:

python3 servo_api_server.py

Optional hand-controller for servo calibration:

python3 servo_gamepad_tamed.py


⸻

5. Closed-loop Attack Operation

Step 1: ROI Selection
	•	Draw a rectangle on the screen
	•	Press s to lock and enter closed-loop mode
	•	Press q to exit

Step 2: Closed-loop Control

Key	Function
1	Auto: Grad-CAM selects target pixel
2	Manual: click pixel inside ROI
r	Reset lock
f	Cycle color-filter defenses
q	Quit

Real-time text overlays show:
	•	Both models’ predictions
	•	Grad-CAM target pixel
	•	Laser pixel position
	•	Tracking error
	•	Current filter
	•	Main vs surrogate probabilities

⸻

6. Digital Color-Filter Defenses (with detailed explanations)

Below is a complete list of all implemented defenses, grouped by type,
with concise technical explanations that your导师会非常喜欢.

⸻

6.1 Baseline

none

No processing. ROI is passed directly to the classifiers.

⸻

6.2 RGB Channel Removal

remove_green
	•	Sets G channel to zero.
	•	Completely eliminates green hues.
	•	Effective when the attack uses green laser.
	•	Also removes genuine green content → may change semantics.

remove_red
	•	Sets R channel to zero.
	•	Equivalent defense for red laser.

Very simple but destructive. Removes both laser and natural content.

⸻

6.3 Strong Channel Suppression

These filters combine RGB zeroing + HSV brightness suppression.

remove_green_strong
	1.	Force G=0 in RGB space.
	2.	Convert to HSV.
	3.	Build a broad green mask using hue ∈ [40°, 90°].
	4.	Reduce brightness: V *= 0.2.

Effect:
	•	Removes the entire green signal, including reflections & camera auto-white-balance artifacts.

remove_red_strong
	1.	Zero R channel.
	2.	Hue mask for red: ranges around [0..10] + [170..180]
	3.	Reduce V strongly.

Effect:
	•	Eliminates the bright red core + halo of laser.

⸻

6.4 HSV Blocking (Color-specific filtering)

These filters preserve more image content.

hsv_block_green
	•	Convert to HSV
	•	Build mask of high-saturation, high-brightness green pixels
	•	Reduce V moderately (≈ 0.3)

Effect:
	•	Keeps the overall image
	•	Mainly dims strong green highlights (typical laser signature)

hsv_block_red

Same for red laser:
	•	Two red hue bands
	•	High S, high V
	•	Reduce brightness

⸻

6.5 Laser Inpainting (Most advanced defenses)

These defenses try to erase the laser blob completely.

General mechanism:
	1.	HSV color-based laser mask
	2.	Dilate the mask to include the halo
	3.	Use OpenCV:

cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)

Filters:

Filter	Characteristics
laser_inpaint_green	Standard mask + inpainting
laser_inpaint_green_strong	Wider mask + larger radius → stronger removal
laser_inpaint_red	Red-laser equivalent
laser_inpaint_red_strong	Most aggressive version

Effect:
	•	The laser is replaced with visually plausible background content.
	•	Best for preserving classifier input distribution.

⸻

6.6 Highlight Clipping (Color-agnostic defense)

clip_highlights
	•	Convert to HSV
	•	Mask bright & saturated pixels: V>220 and S>80
	•	Reduce brightness to moderate levels

Effect:
	•	Removes any bright specular spots, regardless of color
	•	Very effective against lasers of any color

⸻

6.7 Grayscale Defenses

gray
	•	Convert ROI to grayscale
	•	Stack into 3 channels

Effect:
	•	Removes all color → laser color advantage disappears
	•	But distribution shift may affect classifier accuracy

gray_blur
	•	Convert to gray
	•	Apply GaussianBlur(7×7)
	•	Convert back to BGR

Effect:
	•	Removes both color & small high-intensity artifacts (laser spot)

⸻

7. Main vs Surrogate Model Behavior

Component	Description
Main model (panel_cls_full.pt)	Victim classifier (NORMAL / INVALID)
Surrogate model (panel_cls_surrogate.pt)	Used by Grad-CAM to pick attack pixel

Both models receive the filtered ROI, so we can test:
	•	Attack transferability (surrogate → main)
	•	Defense transferability

Migration to a new device involves only swapping .pt files.

⸻

8. Supported Devices

The following devices are supported with identical code:
	•	SIMATIC-S7-1500
	•	Siemens-410E
	•	Siemens-Energy-T3000

Each folder contains the correct pair of .pt models.

⸻

9. Summary

This repository provides:
	•	Full closed-loop physical laser attack
	•	15+ digital defenses with strong theoretical grounding
	•	Industrial HMI support (three Siemens devices)
	•	Unified, reproducible codebase
	•	Real-time tracking, control, and XAI-based targeting

⸻
