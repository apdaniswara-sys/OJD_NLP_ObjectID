# camera_object_count.py
# Real-time Object Detection + Counting using YOLOv8 + OpenCV

from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# 1️⃣ Load model YOLOv8
model = YOLO("yolov8n.pt")  # nano = cepat, ringan

# 2️⃣ Buka kamera (0 = kamera utama)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Tidak bisa membuka kamera.")
    exit()

print("✅ Kamera aktif. Tekan 'q' untuk keluar.")

# 3️⃣ Loop video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # 4️⃣ Jalankan deteksi
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()  # frame dengan bounding box

    # 5️⃣ Ambil daftar nama objek terdeteksi
    names = model.names
    detected_classes = [names[int(cls)] for cls in results[0].boxes.cls]

    # 6️⃣ Hitung jumlah tiap objek
    counts = Counter(detected_classes)

    # 7️⃣ Buat panel kanan untuk menampilkan hasil count
    h, w, _ = annotated_frame.shape
    panel_width = 300
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    # Judul
    cv2.putText(panel, "Object Count", (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Tampilkan hasil count
    y = 80
    for obj, cnt in counts.items():
        text = f"{obj}: {cnt}"
        cv2.putText(panel, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35

    # 8️⃣ Gabungkan panel kanan dengan frame kamera
    combined = np.hstack((annotated_frame, panel))

    # 9️⃣ Tampilkan hasil
    cv2.imshow("OJD — Object Detection & Counting", combined)

    # 10️⃣ Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Bersihkan resource
cap.release()
cv2.destroyAllWindows()

