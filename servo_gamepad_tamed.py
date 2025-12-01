#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_gamepad_tamed.py
USB gamepad control for continuous-rotation servos (GPIO17: yaw, GPIO27: pitch)
Includes: deadzone, low-pass filter, speed gain, fine mode (LT), slew-rate limit,
and online neutral calibration (X/Y).
"""
import os, time, pygame, pigpio

# -- Disable window when running in terminal/SSH --
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ====== Hardware pins ======
PIN_YAW   = 17
PIN_PITCH = 27

# ====== Behaviour parameters (tunable on the fly) ======
NEUTRAL_US       = 1500     # Your "stop" pulse width, e.g. calibrated to 1492/1510 by other scripts
BASE_MAX_DELTA   = 220      # Base limit for max speed (smaller = slower), 300 was too fast
gain             = 0.40     # Speed gain (0.10~1.00), adjustable with D-pad left/right
DEADZONE         = 0.18     # Stick deadzone (0~1), larger = more stable
LPF_ALPHA        = 0.20     # Input low-pass filter coefficient (0.1~0.3)
SLEW_US_PER_SEC  = 500      # Slew-rate limit: max microseconds change per second (smaller = smoother)
REFRESH_HZ       = 30       # Refresh rate, 30 Hz is enough

# ====== Button mapping (typical gamepad) ======
BTN_A      = 0   # Center
BTN_X      = 2   # Neutral -1 us
BTN_Y      = 3   # Neutral +1 us
BTN_LB     = 4   # Invert X (optional)
BTN_RB     = 5   # Invert Y (optional)
BTN_START  = 7   # Center
AXIS_LX    = 0   # Left stick X
AXIS_LY    = 1   # Left stick Y
AXIS_LT    = 2   # Left trigger (some gamepads use 2/5/…, if missing you can use BTN_LB for fine mode)
# D-pad (HAT)
HAT_IDX    = 0

# ====== Utility function ======
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def main():
    global gain, NEUTRAL_US

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("⚠️ No gamepad detected"); return
    js = pygame.joystick.Joystick(0); js.init()
    name = js.get_name()
    print(f"✅ Gamepad connected: {name}")

    pi = pigpio.pi()
    if not pi.connected:
        print("❌ pigpio not running, please start with: sudo pigpiod"); return

    # Invert switches (set to -1 if your left/right are reversed, etc.)
    invert_x = -1    # Left-right inversion: -1 = reversed, 1 = normal
    invert_y =  1

    # Low-pass-filtered input & current output pulse width (for slew-rate limiting)
    x_f, y_f = 0.0, 0.0
    yaw_us   = NEUTRAL_US
    pitch_us = NEUTRAL_US

    clock = pygame.time.Clock()
    print("Controls: left stick for both axes; A/Start = center; LT = fine mode; "
          "D-pad ←/→ adjust speed gain; X/Y fine-tune neutral; Ctrl+C to exit")
    print(f"Initial: NEUTRAL={NEUTRAL_US}us, gain={gain:.2f}, BASE_MAX_DELTA={BASE_MAX_DELTA}us, deadzone={DEADZONE}")

    try:
        while True:
            pygame.event.pump()

            # ---- Read axes ----
            x_raw = js.get_axis(AXIS_LX)
            y_raw = js.get_axis(AXIS_LY)

            # Deadzone
            x_raw = 0.0 if abs(x_raw) < DEADZONE else x_raw
            y_raw = 0.0 if abs(y_raw) < DEADZONE else y_raw

            # Low-pass filter
            x_f = (1.0 - LPF_ALPHA) * x_f + LPF_ALPHA * x_raw
            y_f = (1.0 - LPF_ALPHA) * y_f + LPF_ALPHA * y_raw

            # Fine mode (hold LT to slow down to 0.4x; if LT is an axis, value is usually in -1..1)
            fine = False
            try:
                lt_val = js.get_axis(AXIS_LT)
                # On some gamepads LT is -1 when released and moves towards +1 when pressed
                if lt_val > 0.2:
                    fine = True
            except Exception:
                # If there is no LT axis, use LB as fine mode
                if js.get_button(BTN_LB):
                    fine = True
            gain_eff = gain * (0.40 if fine else 1.0)

            # ---- Compute target pulse widths ----
            max_delta = int(BASE_MAX_DELTA * gain_eff)
            # Note: Y axis is usually negative when pushed up, so we invert it; then apply inversion flags
            tgt_yaw   = clamp(int(NEUTRAL_US + (invert_x * x_f) * max_delta),  900, 2100)
            tgt_pitch = clamp(int(NEUTRAL_US + (invert_y * -y_f) * max_delta), 900, 2100)

            # ---- Slew-rate limiting ----
            max_step = int(SLEW_US_PER_SEC / REFRESH_HZ)
            if tgt_yaw > yaw_us:   yaw_us   = min(yaw_us + max_step, tgt_yaw)
            elif tgt_yaw < yaw_us: yaw_us   = max(yaw_us - max_step, tgt_yaw)
            if tgt_pitch > pitch_us:   pitch_us = min(pitch_us + max_step, tgt_pitch)
            elif tgt_pitch < pitch_us: pitch_us = max(pitch_us - max_step, tgt_pitch)

            # ---- Button actions ----
            # Center
            if js.get_button(BTN_A) or js.get_button(BTN_START):
                yaw_us = pitch_us = NEUTRAL_US

            # Neutral fine-tuning (1 us per press)
            if js.get_button(BTN_X):  # -1 us
                NEUTRAL_US = clamp(NEUTRAL_US - 1, 1200, 1800)
            if js.get_button(BTN_Y):  # +1 us
                NEUTRAL_US = clamp(NEUTRAL_US + 1, 1200, 1800)

            # D-pad (HAT) adjusts gain
            if js.get_numhats() > 0:
                hat = js.get_hat(HAT_IDX)  # (x,y)
                if hat[0] == 1:   # →
                    gain = clamp(gain + 0.02, 0.10, 1.00)
                elif hat[0] == -1: # ←
                    gain = clamp(gain - 0.02, 0.10, 1.00)

            # Optional: LB/RB toggle inversion
            if js.get_button(BTN_LB):
                invert_x *= -1; time.sleep(0.15)
            if js.get_button(BTN_RB):
                invert_y *= -1; time.sleep(0.15)

            # ---- Output ----
            pi.set_servo_pulsewidth(PIN_YAW,   yaw_us)
            pi.set_servo_pulsewidth(PIN_PITCH, pitch_us)

            print(f"\rYaw={yaw_us}us  Pitch={pitch_us}us | NEU={NEUTRAL_US}  gain={gain:.2f}  fine={'ON' if fine else 'OFF'}   ", end="")
            clock.tick(REFRESH_HZ)

    except KeyboardInterrupt:
        print("\nExiting, releasing servos...")
    finally:
        pi.set_servo_pulsewidth(PIN_YAW,   0)
        pi.set_servo_pulsewidth(PIN_PITCH, 0)
        pi.stop()
        pygame.quit()

if __name__ == "__main__":
    main()
