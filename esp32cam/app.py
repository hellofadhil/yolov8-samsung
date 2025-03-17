import cv2 as cv
import numpy as np
import threading

URL = "http://192.168.100.63:81/stream"

cap = cv.VideoCapture(URL)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame = None
frame_lock = threading.Lock()  # Untuk menghindari race condition
frame_count = 0

def capture_frames():
    global frame
    while True:
        ret, temp_frame = cap.read()
        if ret:
            with frame_lock:
                frame = temp_frame.copy()  # Hindari frame berubah saat diproses

thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

def detect_faces(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return frame

while True:
    with frame_lock:
        if frame is None:
            continue
        temp_frame = frame.copy()  # Ambil salinan frame agar tidak terpengaruh thread lain

    frame_count += 1

    if frame_count % 3 == 0:  # Deteksi wajah tiap 3 frame
        temp_frame = detect_faces(temp_frame)

    resized_frame = cv.resize(temp_frame, (320, 240))  # Resize setelah deteksi wajah

    cv.imshow('Face Detection', resized_frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
