# Smart Gas Meter Reader using Raspberry Pi and YOLO

YOLO-based OCR System on Raspberry Pi with Auto Scheduler & Button Trigger

Sistem ini adalah solusi end-to-end untuk membaca angka meteran gas secara otomatis menggunakan Computer Vision (YOLO) di Raspberry Pi.
Dirancang untuk stabil, hemat resource, dan siap terintegrasi dengan API server.

🚀 Fitur Utama

- Capture otomatis berdasarkan jadwal jam tertentu
- Capture manual via tombol (short press)
- Reboot perangkat via tombol (long press)
- Deteksi 8 digit angka menggunakan YOLO
- Preprocessing (CLAHE + Sharpening) untuk meningkatkan akurasi
- Auto pilih frame paling tajam (anti blur)
- Kirim hasil + gambar ke API server

🛠️ Instalasi

Install dependency:

```bash
pip install opencv-python numpy requests ultralytics RPi.GPIO
```

Pastikan:
- Model YOLO tersedia di /best.pt
- API endpoint sudah diisi
- Raspberry Pi memiliki akses kamera

▶️ Cara Menjalankan
```bash
python3 main.py
```
Program akan berjalan terus sebagai monitoring service.

✨ Author

**_Fitriyana Nuril Khaqqi_**

Developed as part of Smart Meter OCR Deployment Project

Computer Vision • Embedded System • IoT Integration
