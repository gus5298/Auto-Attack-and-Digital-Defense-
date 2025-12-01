# Digital-Defense: Closed-loop XAI + Laser Attack & Color-Filter Defenses
**This model is compatible with Siemens Energy T3000. The experiment involves three devices (including Siemens S7-1500, Siemens Energy T3000, etc.).  
Simply replace `panel_cls_full.pt` and `panel_cls_surrogate.pt` to complete the migration.**

This repository contains a **closed-loop physical attack & defense framework** for an industrial vision classifier.  
A camera observes an HMI / panel, a Raspberry Pi drives two servos and a laser pointer, and a PC controls the whole loop using Grad-CAM to aim the laser at the most influential pixels of a **surrogate model** while monitoring a **main (victim) model**.

On top of the basic attack, we implement a set of **digital color filters** as “defenses”: before feeding the ROI into the CNNs, we transform the image to suppress the laser signal while preserving the original content as much as possible.

---

## 1. Repository Structure

Main files in this repo:

- `xai_closedloop_pc.py`  
  PC-side closed-loop controller:
  - loads **main model** `panel_cls_full.pt` and **surrogate model** `panel_cls_surrogate.pt`
  - runs multi-scale Grad-CAM on the surrogate model to select a target pixel inside an ROI
  - detects red/green laser in the full frame
  - computes tracking error and sends step commands to Raspberry Pi
  - applies **digital color filters** to the ROI before classification (Task 3: digital defense)

- `servo_api.py`  
  Raspberry Pi servo + laser API using `pigpio`, implementing **small step “tap” control** for yaw/pitch and laser firing.

- `servo_api_server.py`  
  TCP server on Raspberry Pi:
  - listens on port `50000`
  - receives JSON commands from PC (step_yaw, step_pitch, fire, stop)
  - calls `ServoAPI` accordingly and returns JSON responses.

- `servo_gamepad_tamed.py`  
  Optional: USB gamepad control for continuous servos on the Pi (deadzone, low-pass filter, slew-rate limiting, gain tuning, etc.).

- `viewer_gradmap_judge.py`  
  Offline / debugging viewer:
  - select ROI with the mouse
  - run multi-scale classification
  - optionally overlay Grad-CAM heatmap
  - quickly check **NORMAL vs INVALID** decisions without the full closed loop.

- `model_def.py`  
  Model definitions (`SmallCNN`, `ConvBNReLU`, optional `ResNet18Classifier`) and metadata used by all scripts.

- `panel_cls_full.pt`  
  Main/victim classifier, saved as a whole model (with attached metadata fields like `class_names`, `normal_idx`, `input_size`, `target_layer`, `normalize`, …).

- `panel_cls_surrogate.pt`  
  Surrogate model used for Grad-CAM targeting (e.g. ResNet-18).  
  The **attack** uses this model to pick laser target pixels; we then evaluate transferability to the main model.

---

## 2. Hardware & Setup

- **PC / Laptop**
  - Runs `xai_closedloop_pc.py`
  - Has a USB webcam or built-in camera (tested at 1920×1080)

- **Raspberry Pi**
  - Runs `servo_api_server.py`
  - `pigpio` daemon must be running:  
    ```bash
    sudo pigpiod
    ```
  - GPIO connections:
    - `PIN_YAW = 17`   — yaw servo (horizontal)
    - `PIN_PITCH = 27` — pitch servo (vertical)
    

- **Laser + Servos**
  - Standard hobby servos for pan/tilt
  - Laser pointer driven through a MOSFET (do **not** drive laser directly from GPIO)

- **Network**
  - PC and Raspberry Pi must be on the same network.
  - Set the Pi IP in `xai_closedloop_pc.py`:
    ```python
    PI_IP   = "10.172.153.228"   # change to your Pi's IP
    PI_PORT = 50000
    ```

---

## 3. Software Requirements

On **PC** (Python 3.x):

```bash
pip install torch torchvision opencv-python numpy

On Raspberry Pi:

sudo apt-get install pigpio
pip install pigpio pygame
sudo pigpiod   # start daemon before running scripts

Git LFS is used for large .pt files:

git lfs install
git clone https://github.com/Terminuse-wei/Digital-Defense.git


⸻

4. Running the System

4.1. Raspberry Pi side

cd Digital-Defense
sudo pigpiod                     # if not already running
python3 servo_api_server.py

The Pi will print log messages when it receives step/fire commands from the PC.

4.2. PC side (closed-loop controller)

cd Digital-Defense
python3 xai_closedloop_pc.py

A window XAI Closed-loop (Grad-CAM + Laser) will appear.

Step 1 – ROI selection
	1.	Use the mouse to drag a rectangle around the panel / HMI region in the live camera feed.
	2.	Press s to lock the ROI and start the closed loop.
	3.	Press q at any time to exit.

Step 2 – Closed-loop attack & digital defense
Keyboard controls during the closed-loop phase:
	•	1 – Auto mode
	•	Use the surrogate model’s Grad-CAM once to lock a target pixel inside the ROI.
	•	Target pixel remains fixed afterwards (no drift).
	•	2 – Manual mode
	•	Click inside the ROI to set a custom target pixel.
	•	Useful to mimic “human-chosen” attack positions.
	•	r – Reset lock
	•	Clears the “locked” state so the servos can re-adjust again.
	•	f – Cycle through color filter modes (Task 3: digital defenses).
	•	q – Quit.

During each frame:
	1.	Both main model and surrogate model classify the filtered ROI.
	2.	Based on the main model’s output (NORMAL vs INVALID), the system tells you which color laser to use:
	•	NORMAL (main) → red laser
	•	INVALID (main) → green laser
	3.	The laser spot is detected in the original full frame (unfiltered), the tracking error is computed, and step commands are sent to the Pi until the laser aligns with the target pixel.

Status text at the top of the window shows:
	•	mode / lock state
	•	main model prediction and probabilities
	•	surrogate model prediction and probabilities (if available)
	•	current filter name
	•	target vs. laser pixel coordinates and error

⸻

5. Digital Color Filters (Task 3: “Digital Defense”)

The function apply_color_filter() in xai_closedloop_pc.py implements a set of digital defenses.
These filters are applied only to the ROI before feeding it into the CNNs.
Laser detection always runs on the original, unfiltered frame, so tracking is not affected.

Available modes (in order of cycling with key f):

5.1. none
	•	Baseline (no defense)
	•	The ROI is passed to the models unchanged.

⸻

5.2. Channel removal

remove_green
	•	Set the G channel to zero for the whole image (img[:, :, 1] = 0).
	•	Very strong against green lasers, but all genuine green content in the panel is also removed.

remove_red
	•	Set the R channel to zero (img[:, :, 2] = 0).
	•	Symmetric version for red lasers, removing red content from the scene.

⸻

5.3. Strong channel removal + extra suppression

remove_green_strong
	1.	First apply remove_green (G channel = 0).
	2.	Convert to HSV and create a wide green-like mask.
	3.	For pixels in this mask, aggressively reduce brightness (V channel × 0.2).

Effect:
Even if some residual “pseudo-green” remains due to camera white balance or reflections, these bright greenish areas are further darkened.

remove_red_strong
Symmetric version for red:
	1.	Zero out the R channel.
	2.	Build a mask for both red hue ranges ([0..10] and [170..180]).
	3.	Reduce brightness for masked pixels (V × 0.2).

Effect:
Heavily suppresses both direct red laser spots and strong reddish highlights.

⸻

5.4. HSV highlight suppression

hsv_block_green
	•	In HSV space, select a range of green hues with high saturation & brightness.
	•	Reduce brightness (V) by a factor (≈ 0.3) only in this mask.
	•	Keeps the rest of the image visually similar while dimming saturated green.

hsv_block_red
	•	Same idea for red hues:
	•	two hue intervals around 0° and 180°
	•	Reduce brightness in those regions.

These are moderate, color-specific defenses: they try to keep panel readability while reducing laser contrast.

⸻

5.5. Laser-region inpainting

These filters try to detect the laser region and “erase” it using image inpainting.

laser_inpaint_green
	•	Build a green-laser mask using HSV, slightly wider than detection threshold.
	•	Dilate the mask to cover the glow around the laser spot.
	•	Use cv2.inpaint(..., radius=3) to fill the region from surrounding pixels.

laser_inpaint_green_strong
	•	Use wider green hue range and more dilation, with a larger inpaint radius (5).
	•	More aggressive removal of the whole laser blob and halo.

laser_inpaint_red
	•	Same idea for red:
	•	two hue intervals for red,
	•	dilation, then inpaint with radius 3.

laser_inpaint_red_strong
	•	Wider red range + stronger dilation + larger inpaint radius (5).
	•	Most aggressive red-laser removal.

These are the strongest “surgical” digital defenses: they aim to literally replace the laser pixels with plausible background content.

⸻

5.6. Highlight clipping (color-agnostic)

clip_highlights  / highlight_clip (depending on file)
	•	Convert to HSV, then build a mask for pixels that are:
	•	very bright (V > 220)
	•	and sufficiently saturated (S > 80)
	•	For masked pixels, set brightness to a lower value (e.g. V = 150).

Effect:
Suppresses any very bright, saturated spot, regardless of color – typical laser appearance – while leaving the rest mostly unchanged.

⸻

5.7. Grayscale-based defenses

gray
	•	Convert to grayscale, then replicate to 3 channels.
	•	Completely removes color information, including the color difference between laser and background.
	•	Often effective at killing color-only attacks, but changes the model’s input distribution, so it can also disturb predictions in the no-attack case.

gray_blur
	•	Convert to grayscale.
	•	Apply Gaussian blur (7×7 kernel).
	•	Convert back to 3-channel BGR.

Effect:
Removes both color and some fine-scale structure, so small high-intensity dots (laser spots) are smoothed out.
Again, this is a strong transformation and may also affect normal predictions.

⸻

6. Main vs. Surrogate Model
	•	Main model (victim): panel_cls_full.pt
	•	Used to evaluate whether the panel is NORMAL or INVALID / FAILURE.
	•	Determines which color of laser should be used in the current scenario.
	•	We measure whether attacks / defenses succeed on this model.
	•	Surrogate model: panel_cls_surrogate.pt (e.g. ResNet-18)
	•	Used by Grad-CAM to select the most influential target pixel for the attack.
	•	Filtered ROI is fed to both models with the same defense, so we can study:
	•	attack transferability (surrogate → main),
	•	and defense transferability (a filter that protects the main model may or may not protect the surrogate).

In the UI, both models’ predictions and probabilities are printed each frame.

⸻

7. Known Experimental Use

In our experiments, we applied this framework to industrial devices such as:
	•	Siemens Energy T3000

By running through all filter modes with and without green/red lasers, we summarized which defenses:
	•	keep the no-laser baseline predictions correct,
	•	and successfully block laser-induced misclassification on both main and surrogate models.



