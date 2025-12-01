#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_closedloop_pc.py

PC-side visual closed-loop:
- Main model: panel_cls_full.pt   (original model / victim)
- Surrogate model: panel_cls_full_surrogate.pt (e.g., ResNet18, used for Grad-CAM targeting)

Inside the ROI:
    * Auto mode: use the surrogate model's Grad-CAM to find a hot target pixel (u_target, v_target) once,
                 then keep it fixed (no drifting)
    * Manual mode: you click a point inside the ROI as the target
    * According to the MAIN model's NORMAL / INVALID prediction, automatically choose laser color:
        - NORMAL  -> RED dot, detect red laser
        - INVALID -> GREEN dot, detect green laser
    * Laser spot detection → compute error → send step_yaw / step_pitch to Raspberry Pi,
      so the servos track the target step by step
    * At the same time print both main & surrogate predictions to measure transferability

Task3 (defense):
    * Before feeding into the model, apply digital color filtering on the ROI to suppress laser color,
      without changing the original frame and without affecting laser detection.
    * Press 'f' to switch among different filter modes to compare attack success with / without defense.

Key bindings:
    - ROI stage:
        * Drag mouse to select ROI, press 's' to lock and start closed-loop
        * Press 'q' to quit
    - Closed-loop stage:
        * '1' = Auto mode (Grad-CAM automatically locks target point, locked once and not updated)
        * '2' = Manual mode (click inside ROI to set target point)
        * 'r' = Reset alignment lock (servos can move again, but auto target point stays)
        * 'f' = Switch color filter mode (Task3 defense)
        * 'q' = Quit
"""

import os
import json
import socket
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from model_def import SmallCNN, ConvBNReLU
import torch.serialization
from torch.nn.modules.container import Sequential
from torch.nn import Conv2d, BatchNorm2d, Linear, Dropout2d, ReLU, MaxPool2d, AdaptiveAvgPool2d

# ====== Safe globals list for deserialization ======
torch.serialization.add_safe_globals([
    SmallCNN,
    ConvBNReLU,
    Sequential,
    Conv2d,
    BatchNorm2d,
    Linear,
    Dropout2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
])

# If panel_cls_full.pt / surrogate uses ResNet18Classifier, add a fallback here for compatibility
try:
    from model_def import ResNet18Classifier
    import sys
    torch.serialization.add_safe_globals([ResNet18Classifier])
    sys.modules["__main__"].ResNet18Classifier = ResNet18Classifier
except Exception:
    pass

# ====== Raspberry Pi connection config ======
PI_IP   = "10.172.153.228"   # Change to your Raspberry Pi IP
PI_PORT = 50000

def send_pi_cmd(cmd: dict):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.5)
    try:
        s.connect((PI_IP, PI_PORT))
        s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        data = s.recv(4096)
        txt  = data.decode("utf-8").strip()
        try:
            return json.loads(txt)
        except Exception:
            return {"status": "raw", "raw": txt}
    except Exception as e:
        print("[ERR] send_pi_cmd:", e)
        return {"status": "err", "msg": str(e)}
    finally:
        s.close()

# ====== Some small UI helpers ======
def put_text(img, s, org, color=(0,255,0), scale=0.8, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ====== ROI selector ======
class ROISelector:
    def __init__(self, win):
        self.win = win
        self.dragging = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.roi = None
        self.locked = False

    def on_mouse(self, event, x, y, flags, param):
        if self.locked:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.x0 = self.x1 = x
            self.y0 = self.y1 = y
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            x0, x1 = sorted([self.x0, self.x1])
            y0, y1 = sorted([self.y0, self.y1])
            if x1 - x0 >= 10 and y1 - y0 >= 10:
                self.roi = (x0, y0, x1, y1)

    def draw(self, vis):
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
            put_text(vis, f"ROI[{'lock' if self.locked else 'edit'}]",
                     (x0, max(22, y0 - 8)), (0, 255, 255), 0.7, 2)
        if self.dragging:
            x0, x1 = sorted([self.x0, self.x1])
            y0, y1 = sorted([self.y0, self.y1])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 1)

# ====== Grad-CAM class ======
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model.eval()
        self.target = self._get_module(model, target_layer_name)
        if self.target is None:
            raise ValueError(f"Cannot find layer: {target_layer_name}")
        self.act=None
        self.grad=None
        self.fh = self.target.register_forward_hook(self._fh)
        self.bh = self.target.register_full_backward_hook(self._bh)
    def _get_module(self, root, name):
        m = root
        for n in name.split('.'):
            m = m[int(n)] if n.isdigit() else getattr(m, n, None)
            if m is None:
                return None
        return m
    def _fh(self, m, i, o):
        self.act = o.detach()
    def _bh(self, m, gi, go):
        self.grad = go[0].detach()
    @torch.no_grad()
    def _norm(self, x):
        x = x - x.min()
        return x / (x.max() - x.min() + 1e-6)
    def generate(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits.reshape(1,-1)[0, class_idx]
        score.backward(retain_graph=True)
        A, dA = self.act, self.grad
        w = dA.mean(dim=(2,3), keepdim=True)
        cam = (w * A).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1),
                            size=x.shape[-2:],
                            mode="bilinear",
                            align_corners=False).squeeze(1)
        return self._norm(cam[0].cpu().numpy())

# ====== Model loading (shared by main / surrogate) ======
def load_full_model(device, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    import torch.serialization
    from model_def import SmallCNN
    # Fallback for ResNet18Classifier
    try:
        from model_def import ResNet18Classifier
        import sys
        torch.serialization.add_safe_globals([ResNet18Classifier])
        sys.modules['__main__'].ResNet18Classifier = ResNet18Classifier
    except Exception:
        pass
    torch.serialization.add_safe_globals([SmallCNN])
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    class_names = getattr(model, "class_names", ["normal","network_failure"])
    normal_idx  = int(getattr(model, "normal_idx", 0))
    input_size  = int(getattr(model, "input_size", 224))
    normalize   = getattr(model, "normalize",
                          {"mean":[0.485,0.456,0.406],
                           "std":[0.229,0.224,0.225]})
    # SmallCNN usually uses feat.3; ResNet uses feat.7
    target_layer = getattr(model, "target_layer", "feat.3")
    mean, std = normalize["mean"], normalize["std"]
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    cam = GradCAM(model, target_layer)
    meta = {
        "class_names": class_names,
        "normal_idx":  normal_idx,
        "input_size":  input_size,
        "target_layer": target_layer
    }
    return model, tfm, cam, normal_idx, meta

# ====== Laser spot detection (red / green) — on the full frame ======
def detect_laser_point(bgr_img, color="red"):
    """
    Find the laser spot in the full BGR image (detect over the whole frame).
    Return (x, y) — global image coordinates (used for servo closed-loop).
    Note: this uses the original frame and is not affected by digital filters.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    if color == "red":
        mask1 = cv2.inRange(hsv, (0,150,200),  (10,255,255))
        mask2 = cv2.inRange(hsv, (170,150,200),(180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:  # green
        lower_g = (40, 80, 180)
        upper_g = (90, 255, 255)
        mask = cv2.inRange(hsv, lower_g, upper_g)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy)   # global coordinates (x, y)

# ====== Task3: color filter (digital defense) ======

def apply_color_filter(bgr_img, mode="none"):
    """
    Apply color filtering on a BGR image (only before feeding into the model):

        - "none"                  : no processing (no-defense baseline)
        - "remove_green / red"    : directly zero out the G / R channel for the whole image
        - "hsv_block_red / green" : suppress bright red / green in HSV space
        - "gray"                  : convert to 3-channel grayscale, completely remove color information
        - "remove_green_strong /
           remove_red_strong"     : remove G/R, then further suppress bright regions of that color
        - "clip_highlights"       : suppress highly bright and saturated points
        - "laser_inpaint_green /
           laser_inpaint_red"     : inpaint to remove green / red laser regions
        - "laser_inpaint_green_strong /
           laser_inpaint_red_strong" : stronger laser inpaint
        - "gray_blur"             : grayscale + blur
    """
    if mode == "none":
        return bgr_img

    # ★ Directly zero G / R channel
    if mode == "remove_green":
        img = bgr_img.copy()
        img[:, :, 1] = 0        # BGR: channel 1 is G
        return img

    if mode == "remove_red":
        img = bgr_img.copy()
        img[:, :, 2] = 0        # BGR: channel 2 is R
        return img

    # ★ Convert to pure grayscale (then replicate to 3 channels), completely removing color differences (including laser color)
    if mode == "gray":
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.merge([gray, gray, gray])
        return gray3

    img = bgr_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Suppress bright red
    if mode == "hsv_block_red":
        # Red: two hue ranges
        lower1 = np.array([0, 120, 180], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 120, 180], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        # Only decrease brightness (V channel) for these regions (do not set to zero), to keep readability
        v = hsv[:, :, 2].astype(np.float32)
        v[mask > 0] *= 0.3
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    # Suppress bright green
    if mode == "hsv_block_green":
        lower_g = np.array([40, 120, 180], dtype=np.uint8)
        upper_g = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_g, upper_g)
        v = hsv[:, :, 2].astype(np.float32)
        v[mask > 0] *= 0.3
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    # ★ Stronger remove_green_strong: first remove_green, then further suppress "green-looking bright regions"
    if mode == "remove_green_strong":
        # First zero out the G channel
        img = bgr_img.copy()
        img[:, :, 1] = 0

        hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # A relatively wide green mask
        lower_g2 = np.array([40, 50, 80], dtype=np.uint8)
        upper_g2 = np.array([90, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv2, lower_g2, upper_g2)
        v2 = hsv2[:, :, 2].astype(np.float32)
        v2[mask2 > 0] *= 0.2   # further suppress brightness heavily
        v2 = np.clip(v2, 0, 255).astype(np.uint8)
        hsv2[:, :, 2] = v2
        img2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        return img2

    # ★ Symmetric version for red: remove_red_strong
    if mode == "remove_red_strong":
        img = bgr_img.copy()
        img[:, :, 2] = 0

        hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Two red ranges
        lower1 = np.array([0, 50, 80], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 50, 80], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv2, lower1, upper1)
        mask2 = cv2.inRange(hsv2, lower2, upper2)
        mask2_all = cv2.bitwise_or(mask1, mask2)

        v2 = hsv2[:, :, 2].astype(np.float32)
        v2[mask2_all > 0] *= 0.2
        v2 = np.clip(v2, 0, 255).astype(np.uint8)
        hsv2[:, :, 2] = v2
        img2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        return img2

    if mode == "clip_highlights":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        # Only look for "very bright and very saturated" points (typical laser)
        mask = (s > 150) & (v > 200)
        mask = mask.astype(np.uint8)  # 0/1

        v = v.astype(np.float32)
        v[mask == 1] *= 0.1  # reduce brightness of these points to 10% of original
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv[:, :, 2] = v
        img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img2

    # B3: precise mask + inpaint for green laser (strongest green defense)
    if mode == "laser_inpaint_green":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Typical range of green laser (similar to detection range but a bit wider)
        lower_g = np.array([40, 60, 150], dtype=np.uint8)
        upper_g = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_g, upper_g)

        # Slight dilation (cover halo around laser spot)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Use inpaint to remove laser region
        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    # ★ Symmetric version for red: laser_inpaint_red
    if mode == "laser_inpaint_red":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 60, 150], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 60, 150], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    # B1: stronger green laser inpaint (wider range + larger inpaint radius)
    if mode == "laser_inpaint_green_strong":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Wider green range than laser_inpaint_green
        lower_g = np.array([35, 40, 140], dtype=np.uint8)
        upper_g = np.array([100, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_g, upper_g)

        # Dilate multiple times, to cover halo around laser
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Larger inpaint radius
        inpainted = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        return inpainted

    # ★ Symmetric version for red: laser_inpaint_red_strong
    if mode == "laser_inpaint_red_strong":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 40, 140], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 40, 140], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        inpainted = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        return inpainted

    # B2: highlight clipping (for both red/green — any extremely bright and saturated region is darkened)
    if mode == "highlight_clip":
        img = bgr_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        # Condition: very bright + reasonably high saturation
        mask = (v > 220) & (s > 80)
        v = v.astype(np.float32)
        v[mask] = 150  # directly clamp those regions to a lower brightness
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    # B3: grayscale (keep only structure, drop color) — this gray branch is actually unreachable,
    # because we already returned earlier for mode == "gray"; kept here for completeness.
    if mode == "gray":
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    # B4: grayscale + blur (further remove small high-light structures)
    if mode == "gray_blur":
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    return bgr_img

# ====== Multi-scale + Grad-CAM, return target point and classification (generic) ======
def gradcam_target_pixel(model, tfm, cam, device, roi_bgr,
                         normal_idx=0, normal_thr=0.6,
                         filter_mode="none"):
    """
    Multi-scale center cropping + classification + Grad-CAM on the ROI.
    ★ Here we first apply the color filter (Task3) on roi_bgr before feeding into the model.
    """
    # 1) Apply color filter first (digital defense, only change model input)
    roi_bgr_filtered = apply_color_filter(roi_bgr, mode=filter_mode)

    Hroi, Wroi = roi_bgr_filtered.shape[:2]
    rgb = cv2.cvtColor(roi_bgr_filtered, cv2.COLOR_BGR2RGB)
    cx, cy = Wroi//2, Hroi//2
    scales = [1.00,0.85,0.70,0.55,0.45]
    best = {"p1": -1}
    best_x = None
    for s in scales:
        ww = int(Wroi * s)
        hh = int(Hroi * s)
        x0 = max(0, cx - ww//2); x1 = min(Wroi, cx + ww//2)
        y0 = max(0, cy - hh//2); y1 = min(Hroi, cy + hh//2)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        x = tfm(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, 1)[0].cpu().numpy()
        if prob[1] > best["p1"]:
            best = {
                "p1": prob[1],
                "p0": prob[0],
                "pred": int(np.argmax(prob)),
                "scale": s,
                "p_top": float(prob[int(np.argmax(prob))]),
            }
            best_x = x
    if best_x is None:
        x = tfm(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, 1)[0].cpu().numpy()
        best = {
            "p1": prob[1],
            "p0": prob[0],
            "pred": int(np.argmax(prob)),
            "scale": 1.0,
            "p_top": float(prob[int(np.argmax(prob))]),
        }
        best_x = x
    pred_idx = best["pred"]
    is_normal = (pred_idx == normal_idx) and best["p_top"] >= normal_thr
    heat_small = cam.generate(best_x, class_idx=pred_idx)
    heat = cv2.resize(heat_small, (Wroi,Hroi), interpolation=cv2.INTER_LINEAR)
    max_idx = np.argmax(heat)
    v, u = divmod(max_idx, Wroi)
    prob_vec = np.array([best["p0"], best["p1"]], dtype=np.float32)
    return heat, (u,v), prob_vec, pred_idx, is_normal, best["scale"], best["p_top"]

# ====== Main program ======
MODEL_PATH_MAIN      = "panel_cls_full.pt"          # original victim model
MODEL_PATH_SURROGATE = "panel_cls_surrogate.pt"     # surrogate model (ResNet18)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Main model ----
    model_main, tfm_main, cam_main, normal_idx_main, meta_main = load_full_model(device, MODEL_PATH_MAIN)
    print("[INFO] Main model loaded:", meta_main)

    # ---- Surrogate model (for Grad-CAM) ----
    surrogate_available = False
    try:
        model_surr, tfm_surr, cam_surr, normal_idx_surr, meta_surr = load_full_model(device, MODEL_PATH_SURROGATE)
        surrogate_available = True
        print("[INFO] Surrogate model loaded for Grad-CAM targeting:", meta_surr)
    except FileNotFoundError:
        model_surr = model_main
        tfm_surr   = tfm_main
        cam_surr   = cam_main
        normal_idx_surr = normal_idx_main
        meta_surr  = meta_main
        print("[WARN] Surrogate model panel_cls_surrogate.pt not found, Grad-CAM will reuse main model.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    win = "XAI Closed-loop (Grad-CAM + Laser)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    selector = ROISelector(win)

    # Modes: auto / manual
    mode = "auto"
    manual_target_uv = None   # target point (u,v) in ROI for manual mode
    auto_target_uv   = None   # first Grad-CAM target (u,v) in ROI for auto mode

    roi_box = None  # (x0,y0,x1,y1)

    NORMAL_THR  = 0.60
    PIX_ERR_THR = 15      # alignment threshold (in pixels)
    LOCK_FRAMES = 2       # number of consecutive frames within threshold to treat as locked

    INVERT_YAW   = -1     # adjust according to servo wiring
    INVERT_PITCH = -1

    # ★ Task3: filter modes
    # Filter modes: no defense + HSV defense + strong remove_green + etc.
    # Switch order: no defense → mild green defense → mild red defense → only blur green → only blur red → strongest remove_green
    # Final list for Task3 (no defense + HSV defense + strong remove + grayscale)
    FILTER_MODES = [
        "none",

        "hsv_block_green",
        "hsv_block_red",

        "remove_green",
        "remove_red",

        "remove_green_strong",
        "remove_red_strong",

        "clip_highlights",

        "laser_inpaint_green",
        "laser_inpaint_red",

        "laser_inpaint_green_strong",
        "laser_inpaint_red_strong",

        "gray",
        "gray_blur",
    ]
    filter_idx = 0
    current_filter_mode = FILTER_MODES[filter_idx]

    # Closed-loop state
    stable_frames = 0
    locked_aim = False

    # Mouse callback: ROI + manual target
    def mouse_cb(event, x, y, flags, param):
        nonlocal manual_target_uv, roi_box, mode
        selector.on_mouse(event, x, y, flags, param)
        if selector.locked and roi_box is not None and mode == "manual":
            if event == cv2.EVENT_LBUTTONDOWN:
                x0,y0,x1,y1 = roi_box
                if x0 <= x < x1 and y0 <= y < y1:
                    manual_target_uv = (x - x0, y - y0)
                    print(f"[INFO] Manual target point = {manual_target_uv}")
    cv2.setMouseCallback(win, mouse_cb)

    # -------- ROI selection --------
    print("========== ROI selection ==========")
    print("Drag mouse to select ROI, press 's' to start, 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        selector.draw(vis)
        H,W = vis.shape[:2]
        put_text(vis, "Drag ROI, press 's' to start", (10,30), (0,255,0), 0.7,2)
        put_text(vis, "Press 'q' to quit", (10,H-15), (200,200,200), 0.6,1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if k == ord('s'):
            if selector.roi is None:
                print("[WARN] Please drag to select an ROI before pressing 's'")
                continue
            roi_box = selector.roi
            selector.locked = True
            print("[INFO] ROI locked:", roi_box)
            break

    print("========== Closed-loop started ==========")
    print("1 = Auto mode (Grad-CAM, lock target once using surrogate model)")
    print("2 = Manual mode (click inside ROI to set target)")
    print("r = Reset LOCKED state (allow re-adjustment)")
    print("f = Switch color filter mode (Task3)")
    print("q = Quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            H,W = frame.shape[:2]
            x0,y0,x1,y1 = roi_box
            x0 = clamp(x0,0,W-2); x1 = clamp(x1,1,W-1)
            y0 = clamp(y0,0,H-2); y1 = clamp(y1,1,H-1)
            roi_bgr = frame[y0:y1, x0:x1].copy()
            if roi_bgr.size == 0:
                print("[WARN] ROI size = 0"); continue

            # 1) Use surrogate model for Grad-CAM + classification (decide target point, with color filter)
            try:
                heat_surr, (u_auto,v_auto), prob_surr, pred_surr, is_normal_surr, scale_surr, p_top_surr = \
                    gradcam_target_pixel(
                        model_surr, tfm_surr, cam_surr, device, roi_bgr,
                        normal_idx=normal_idx_surr,
                        normal_thr=NORMAL_THR,
                        filter_mode=current_filter_mode
                    )
            except Exception as e:
                print("[ERR] surrogate Grad-CAM error:", e)
                traceback.print_exc()
                continue

            # Main model also does one classification (with the same color filter)
            try:
                _, _, prob_main, pred_main, is_normal_main, _, p_top_main = \
                    gradcam_target_pixel(
                        model_main, tfm_main, cam_main, device, roi_bgr,
                        normal_idx=normal_idx_main,
                        normal_thr=NORMAL_THR,
                        filter_mode=current_filter_mode
                    )
            except Exception as e:
                print("[ERR] main model inference error:", e)
                traceback.print_exc()
                continue

            cls_name_main = meta_main["class_names"][pred_main] if "class_names" in meta_main else str(pred_main)

            # NORMAL / INVALID → choose red or green laser (based on MAIN model)
            if is_normal_main:
                laser_color = "red"
                laser_hint = "NORMAL (main) → Please use RED laser"
            else:
                laser_color = "green"
                laser_hint = "INVALID (main) → Please use GREEN laser"

            # 2) Choose target point: auto / manual (target from surrogate CAM)
            if mode == "auto":
                if auto_target_uv is None:
                    auto_target_uv = (u_auto, v_auto)
                    print(f"[INFO] Auto mode locked target (from surrogate Grad-CAM): {auto_target_uv}")
                u_t, v_t = auto_target_uv
            else:
                if manual_target_uv is not None:
                    u_t, v_t = manual_target_uv
                else:
                    u_t, v_t = (u_auto, v_auto)

            # Target point in global coordinates (with respect to full frame)
            x_t = x0 + u_t
            y_t = y0 + v_t

            # 3) Detect laser spot in the full frame (not limited to ROI)
            laser_xy = detect_laser_point(frame, color=laser_color)

            # Visualize ROI + surrogate Grad-CAM (overlay on unfiltered roi_bgr for easier inspection)
            roi_vis = roi_bgr.copy()
            heat_u8 = (np.clip(heat_surr,0,1)*255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            roi_vis = cv2.addWeighted(roi_vis, 0.5, heat_color, 0.5, 0)
            # draw target point (blue, ROI coordinates)
            cv2.circle(roi_vis, (int(u_t), int(v_t)), 6, (255,0,0), 2)

            status = ""
            if laser_xy is not None:
                x_l, y_l = laser_xy  # global

                # if laser is inside ROI, also draw a circle there for visualization
                u_l = x_l - x0
                v_l = y_l - y0
                in_roi = (0 <= u_l < (x1-x0)) and (0 <= v_l < (y1-y0))
                if in_roi:
                    cv2.circle(roi_vis, (int(u_l), int(v_l)), 6, (0,0,255), 2)

                # compute error in global coordinates
                err_u = x_t - x_l
                err_v = y_t - y_l
                status = f"target=({x_t:.0f},{y_t:.0f}) laser=({x_l},{y_l}) err=({err_u:.0f},{err_v:.0f})"
            else:
                err_u = err_v = None
                status = "Laser NOT detected"
                stable_frames = 0  # do not accumulate if laser is missing

            # Put ROI back into full frame
            vis = frame.copy()
            h_roi = y1 - y0
            w_roi = x1 - x0
            if roi_vis.shape[0] != h_roi or roi_vis.shape[1] != w_roi:
                roi_vis = cv2.resize(roi_vis, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
            vis[y0:y1, x0:x1] = roi_vis
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)

            # If laser detected, also draw a circle in full-frame coordinates (for visualization)
            if laser_xy is not None:
                cv2.circle(vis, (int(laser_xy[0]), int(laser_xy[1])), 6, (0,0,255), 2)

            # 4) Servo closed-loop + alignment lock
            if laser_xy is not None and err_u is not None and err_v is not None:
                if abs(err_u) <= PIX_ERR_THR and abs(err_v) <= PIX_ERR_THR:
                    stable_frames += 1
                else:
                    stable_frames = 0
                if not locked_aim and stable_frames >= LOCK_FRAMES:
                    locked_aim = True
                    print("[INFO] Target aligned, entering LOCKED state (stop stepping)")

                step_yaw = 0
                step_pitch = 0
                if not locked_aim:
                    if abs(err_u) > PIX_ERR_THR:
                        dir_yaw = +1 if err_u > 0 else -1
                        step_yaw = INVERT_YAW * dir_yaw
                    if abs(err_v) > PIX_ERR_THR:
                        dir_pitch = +1 if err_v > 0 else -1
                        step_pitch = INVERT_PITCH * dir_pitch
                    if step_yaw != 0:
                        send_pi_cmd({"cmd": "step_yaw", "dir": int(step_yaw)})
                    if step_pitch != 0:
                        send_pi_cmd({"cmd": "step_pitch", "dir": int(step_pitch)})
            else:
                stable_frames = 0

            # 5) Status text (print both main & surrogate predictions)
            lock_flag = "LOCKED" if locked_aim else "TRACKING"
            text1 = f"Mode={mode.upper()} | {lock_flag} | MAIN: {cls_name_main} P0={prob_main[0]:.2f} P1={prob_main[1]:.2f}"
            put_text(vis, text1, (10,30), (0,255,0) if is_normal_main else (0,0,255), 0.6,2)

            if surrogate_available:
                cls_name_surr = meta_surr["class_names"][pred_surr] if "class_names" in meta_surr else str(pred_surr)
                text2 = f"SURR: {cls_name_surr} P0={prob_surr[0]:.2f} P1={prob_surr[1]:.2f}"
                put_text(vis, text2, (10,55), (255,255,0), 0.6,2)
                y_hint = 80
            else:
                y_hint = 55

            put_text(vis, laser_hint, (10,y_hint), (0,255,255), 0.6,2)
            put_text(vis, status, (10,y_hint+25), (255,255,0), 0.6,2)

            # ★ Show current filter mode (for Task3)
            put_text(vis, f"Filter={current_filter_mode}", (10,y_hint+50), (150,255,150), 0.6,2)

            H,W = vis.shape[:2]
            put_text(vis, "1=auto, 2=manual(click in ROI), r=reset lock, f=filter, q=quit",
                     (10,H-15), (200,200,200), 0.6,1)

            cv2.imshow(win, vis)
            k = cv2.waitKey(50) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('1'):
                mode = "auto"
                manual_target_uv = None
                auto_target_uv = None  # next frame's Grad-CAM will pick a new target
                locked_aim = False
                stable_frames = 0
                print("[INFO] Switched to AUTO mode (next Grad-CAM will re-lock a target point)")
            elif k == ord('2'):
                mode = "manual"
                manual_target_uv = None
                locked_aim = False
                stable_frames = 0
                print("[INFO] Switched to MANUAL mode, please click inside ROI to set target")
            elif k == ord('r'):
                locked_aim = False
                stable_frames = 0
                print("[INFO] Reset LOCKED state, servos can adjust again (target point unchanged)")
            elif k == ord('f'):
                # Switch color filter mode
                filter_idx = (filter_idx + 1) % len(FILTER_MODES)
                current_filter_mode = FILTER_MODES[filter_idx]
                print("[INFO] Switched color filter mode:", current_filter_mode)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()