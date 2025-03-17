import cv2 as cv
from ultralytics import YOLO

URL = "http://192.168.100.63:81/stream"

# model person detection
model = YOLO("./model/yolov8/ckpts/best.pt")

# model no mask, NO-Safety Vest, NO-Hardhat, Person
# model = YOLO("./model/YOLOv8-custom-object-detection/PPE-cutom-object-detection-with-YOLOv8/ppe.pt")


cap = cv.VideoCapture(URL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dalam frame
    results = model(frame)

    # Proses hasil deteksi
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} ({box.conf[0] * 100:.2f}%)"
            
            # Gambar kotak dan teks
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(label)  # Cetak hasil deteksi ke terminal

    # Tampilkan frame
    cv.imshow("YOLO Detection", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
