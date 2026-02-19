#!/usr/bin/env python3
import os
import time
import threading
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import cv2
import numpy as np
import requests
import RPi.GPIO as GPIO
from ultralytics import YOLO

# === Load Model YOLO ===
model = YOLO("/best.pt")

# === Konfigurasi Tombol ===
BUTTON_PIN = 23
DEBOUNCE_SEC = 0.05
SHORT_PRESS_MAX = 1.0
LONG_PRESS_MIN = 3.0

# === Lock agar kamera tidak diakses bersamaan (auto vs tombol) ===
cam_lock = threading.Lock()

# === Kamera jendela (global) ===
window_cap = None
window_pump_stop = threading.Event()
window_pump_thread = None

# === Preprocessing ===
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(clahe_applied, -1, kernel)
    return sharpened

# === Inference YOLO ===
def run_inference(filepath):
    results = model(filepath)[0]
    predictions = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        x_center = float(box.xywh[0][0])
        predictions.append({"class": label, "x": x_center})

    sorted_digits = sorted(predictions, key=lambda x: x["x"])
    digit_classes = [pred["class"] for pred in sorted_digits[:8]]
    number = "".join(digit_classes).zfill(8)
    return number

# === Kirim Data ke API ===
def send_to_api(temp_path, number):
    integer_part = number[:-3]
    decimal_part = number[-3:]
    real_num = f"{integer_part}.{decimal_part}"

    n_meter_float = float(
        real_num
    )
    print(f"Prediksi meteran: {n_meter_float} | Raw: {real_num}")

    data = {
        "put your data here"
    }

    files = {"file": open(temp_path, "rb")}
    try:
        r = requests.post("https://put your API address here", data=data, files=files, timeout=20)
        print("Status:", r.status_code)
        print("Response:", r.text[:500])
    except Exception as e:
        print("Gagal kirim ke API:", e)
    finally:
        files["file"].close()

# === Open Camera ===
def open_camera(width=640, height=480, fps=30):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def brightness_mean(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

def warmup_and_pick_sharpest(cap, warmup_ms=800, burst_ms=500, step_ms=30):
    t_end = time.time() + (warmup_ms / 1000.0)
    last = None
    while time.time() < t_end:
        ret, f = cap.read()
        if ret:
            last = f
        time.sleep(0.02)
    best = last
    best_score = -1.0
    t_end_burst = time.time() + (burst_ms / 1000.0)
    while time.time() < t_end_burst:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        sharp = variance_of_laplacian(f)
        bright = brightness_mean(f)
        score = sharp if bright >= 10.0 else sharp * 0.1
        if score > best_score:
            best = f
            best_score = score
        time.sleep(step_ms / 1000.0)
    return best

# === Fungsi agar kamera mengambil foto 1x ===
def capture_once(save_folder, tag="AUTO"):
    global window_cap
    with cam_lock:
        use_cap = window_cap if window_cap is not None else open_camera(640, 480, fps=30)
        opened_here = (use_cap is not window_cap)

        if use_cap is None or not use_cap.isOpened():
            print(f"[{tag}] Gagal membuka kamera!")
            return False

        try:
            frame = warmup_and_pick_sharpest(
                use_cap,
                warmup_ms=400 if not opened_here else 800,
                burst_ms=400,
                step_ms=30
            )
            if frame is None:
                print(f"[{tag}] Gagal mendapatkan frame")
                return False

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(save_folder, f"MTR_{ts}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"[{tag}] Foto tersimpan: {filepath}")

            try:
                _ = preprocessing(frame)         
                number = run_inference(filepath) 
                send_to_api(filepath, number)
            except Exception as e:
                print(f"[{tag}] Gagal infer/kirim:", e)
                return False

            return True

        finally:
            if opened_here and use_cap is not None:
                use_cap.release()
                print(f"[{tag}] Kamera dimatikan (capture-only)")

# === Thread Tombol ===
def tombol_loop(save_folder):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print("[BTN] Tombol siap: singkat=jepret+API, lama=reboot")
    press_start = None
    last_state = GPIO.input(BUTTON_PIN)
    try:
        while True:
            state = GPIO.input(BUTTON_PIN)
            if state == GPIO.LOW and last_state == GPIO.HIGH:
                press_start = time.time()
                time.sleep(DEBOUNCE_SEC)
            if state == GPIO.HIGH and last_state == GPIO.LOW and press_start is not None:
                dt = time.time() - press_start
                if dt < SHORT_PRESS_MAX:
                    print(f"[BTN] Tekan singkat ({dt:.2f}s) -> jepret+API")
                    capture_once(save_folder, tag="BTN")
                elif dt >= LONG_PRESS_MIN:
                    print(f"[BTN] Tekan lama ({dt:.2f}s) -> reboot")
                    os.system("sudo reboot")
                press_start = None
            last_state = state
            time.sleep(0.01)
    except Exception as e:
        print("[BTN ERR]", e)
    finally:
        GPIO.cleanup()

# === Konfigurasi jam & waktu ===
capture_hours = [7, 8, 9] 
WINDOW_BEFORE_MIN = 5
WINDOW_AFTER_MIN = 5

def in_window_for_hour(now: datetime, hour: int) -> bool:
    """ True jika now berada di [hour:00 - 5m, hour:00 + 5m] """
    t = now.hour * 60 + now.minute
    center = hour * 60
    return (center - WINDOW_BEFORE_MIN) <= t <= (center + WINDOW_AFTER_MIN)

def any_window_active(now: datetime) -> bool:
    for h in capture_hours:
        if in_window_for_hour(now, h):
            return True
    return False

def start_window_camera():
    global window_cap, window_pump_stop, window_pump_thread
    if window_cap is not None:
        return
    print("[WIN] Mengaktifkan kamera jendela (-5..+5 menit)")
    window_cap = open_camera(640, 480, fps=30)
    if window_cap is None or not window_cap.isOpened():
        print("[WIN] Gagal membuka kamera jendela")
        window_cap = None
        return
    with cam_lock:
        for _ in range(10):
            window_cap.read()
            time.sleep(0.02)
    window_pump_stop.clear()
    window_pump_thread = threading.Thread(target=_window_pump_loop, daemon=True)
    window_pump_thread.start()

def stop_window_camera():
    global window_cap, window_pump_stop, window_pump_thread
    if window_cap is None:
        return
    print("[WIN] Mematikan kamera jendela (keluar dari jendela waktu)")
    window_pump_stop.set()
    if window_pump_thread is not None:
        window_pump_thread.join(timeout=1.0)
    with cam_lock:
        try:
            window_cap.release()
        except Exception:
            pass
    window_cap = None
    window_pump_thread = None

def _window_pump_loop():
    while not window_pump_stop.is_set():
        got = cam_lock.acquire(blocking=False)
        if got:
            try:
                if window_cap is not None:
                    window_cap.read()
            finally:
                cam_lock.release()
        time.sleep(0.2)

def main():
    print("START METERAN", datetime.now())
    save_folder = "foto_otomatis"
    os.makedirs(save_folder, exist_ok=True)

    tombol_thread = threading.Thread(target=tombol_loop, args=(save_folder,), daemon=True)
    tombol_thread.start()

    captured_today = set()
    print("Program monitoring dimulai.")
    print(f"Jam pengambilan gambar: {capture_hours} (kamera ON hanya -5..+5 menit di sekitar jam tersebut)")

    while True:
        now = datetime.now()

        if any_window_active(now):
            if window_cap is None:
                start_window_camera()
        else:
            if window_cap is not None:
                stop_window_camera()

        if (now.minute == 0 and now.second < 10) and (now.hour in capture_hours) and (now.hour not in captured_today):
            ok = capture_once(save_folder, tag="AUTO")
            if ok:
                captured_today.add(now.hour)
                print(f"[AUTO] Gambar berhasil diambil pada {now.hour:02d}:{now.minute:02d}")
            else:
                print("[AUTO] Pengambilan gagal (akan tetap dicoba sekali lagi jika masih dalam detik awal)")

        if now.hour == 0 and now.minute == 0 and now.second < 5:
            captured_today.clear()
            print("[SCHED] Status capture direset untuk hari baru")

        time.sleep(1)

if __name__ == '__main__':
    main()
