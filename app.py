import cv2 as cv
import torch
from ultralytics import YOLO

# Load model YOLOv8 yang sudah dilatih
model = YOLO("./model/yolov8/ckpts/best.pt")  # Sesuaikan path modelmu

# Gunakan kamera laptop
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 420)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 340)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dalam frame
    results = model(frame)

    # Gambar bounding box pada frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Tampilkan hasil
    cv.imshow("YOLO Face Detection", frame)

    if cv.waitKey(1) == ord('q'):
        break

# Bersihkan resource
cap.release()
cv.destroyAllWindows()
