"""
===========================================================
OJD PROJECT — NLP & OBJECT IDENTIFICATION
Author  : Muizz Amin
Purpose : Latihan OJD mengenai Natural Language Processing
           dan Object Detection (YOLOv8 Pretrained)
===========================================================
"""

# ===============================
# 🧠 1. Natural Language Processing (NLP)
# ===============================
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

print("\n🧠 [NLP SECTION] — Sentiment Analysis Demo")

# Download data (hanya pertama kali)
nltk.download('vader_lexicon', quiet=True)

# Contoh teks
texts = [
    "The new forklift system works perfectly and improves efficiency!",
    "The process is too slow, and operators are frustrated.",
    "Overall, the safety improvements are decent but can be better."
]

# Inisialisasi analisis sentimen
sia = SentimentIntensityAnalyzer()

# Analisis setiap kalimat
for t in texts:
    score = sia.polarity_scores(t)
    sentiment = (
        "Positive" if score['compound'] > 0.05 else
        "Negative" if score['compound'] < -0.05 else
        "Neutral"
    )
    print(f"🔹 Text: {t}\n   → Sentiment: {sentiment} ({score['compound']})")

# ===============================
# 🎯 2. Object Identification (YOLOv8)
# ===============================
print("\n🎯 [OBJECT IDENTIFICATION SECTION] — YOLOv8 Demo")

from ultralytics import YOLO
import cv2

# Load model YOLOv8 pretrained
model = YOLO("yolov8n.pt")  # 'n' = nano version (ringan dan cepat)

# Contoh: gunakan gambar sample (bisa ganti file sendiri di folder data)
img_path = "data/sample.jpg"

try:
    results = model(img_path, show=False)
    print("✅ Deteksi objek berhasil dilakukan.")

    # Tampilkan hasil deteksi di console
    for r in results:
        boxes = r.boxes
        print(f"Jumlah objek terdeteksi: {len(boxes)}")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"  - {label} ({conf:.2f})")

    # Simpan hasil anotasi ke folder results
    save_path = "results/detection_output.jpg"
    results[0].save(filename=save_path)
    print(f"🖼️ Hasil deteksi disimpan ke: {save_path}")

except Exception as e:
    print("⚠️ Tidak dapat menemukan file 'data/sample.jpg'.")
    print("   → Tambahkan 1 gambar ke folder data/ dengan nama 'sample.jpg'")
    print(f"   Detail error: {e}")

# ===============================
# ✅ Selesai
# ===============================
print("\n✅ Program selesai dijalankan dengan sukses.")
