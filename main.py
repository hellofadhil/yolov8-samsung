import cv2 as cv
from ultralytics import YOLO

# Load model YOLO (bisa ganti model yang lebih ringan seperti yolov8n.pt)
model = YOLO("./model/yolov8/ckpts/last.pt")  

# Gunakan kamera laptop
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Sesuaikan resolusi agar lebih cepat
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)  # Set FPS untuk kelancaran video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perkecil resolusi frame untuk mempercepat pemrosesan
    frame = cv.resize(frame, (640, 480))

    # Deteksi objek dalam frame dengan mode streaming untuk lebih cepat
    results = model.predict(frame, stream=True)

    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.5:  # Hanya tampilkan deteksi dengan confidence > 50%
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]} ({box.conf[0] * 100:.2f}%)"
                
                # Gambar kotak dan teks
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # print(label)  # Cetak hasil deteksi ke terminal

    # Tampilkan hasil deteksi
    cv.imshow("YOLO Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
