import cv2
import tensorflow as tf
import numpy as np

# Load model
model_dir = "./saved_model"
detect_fn = tf.saved_model.load(model_dir)

# Load label map
category_index = {
    # 1: 'person',
    # 73: 'laptop',
    74: 'mouse',
    # 75: 'keyboard',
    # 77: 'tv/monitor'
}

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Deteksi objek
    detections = detect_fn(input_tensor)

    # Ambil informasi deteksi
    num_detections = int(detections.pop('num_detections'))
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    # Gambar kotak di sekitar objek yang terdeteksi
    height, width, _ = frame.shape
    for i in range(num_detections):
        if detection_scores[i] > 0.5:  # Ambang batas kepercayaan 50%
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)

            label = category_index.get(detection_classes[i], "Unknown")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
