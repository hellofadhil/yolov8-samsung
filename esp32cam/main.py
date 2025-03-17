import cv2 as cv
import numpy as np

# Camera URL
URL = "http://192.168.100.63:81/stream"  # Replace with your ESP32 camera IP

# Initialize camera stream
cap = cv.VideoCapture(URL)

# Check if the camera is opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Load pre-trained classifiers for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set smaller resolution for faster performance (640x480 is a balanced resolution)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Define vibrant colors for drawing
face_color = (255, 255, 0)  # Bright yellow for face bounding box
highlight_color = (0, 255, 255)  # Neon yellow for highlighted effect
label_color = (255, 255, 255)  # White for text labels
text_bg_color = (0, 0, 0)  # Black background for text
font = cv.FONT_HERSHEY_SIMPLEX

# Draw label with background for visibility
def draw_label(frame, label, position, color=(255, 255, 255)):
    """Draw label with background"""
    text_size = cv.getTextSize(label, font, 0.8, 2)[0]
    text_x = position[0]
    text_y = position[1] - 10

    # Draw a background for text
    cv.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), text_bg_color, cv.FILLED)
    cv.putText(frame, label, (text_x, text_y), font, 0.8, color, 2, cv.LINE_AA)

# Apply a smooth gradient effect for bounding boxes
def draw_gradient_box(frame, x, y, w, h):
    """Create a gradient bounding box effect for face"""
    for i in range(3):
        cv.rectangle(frame, (x + i, y + i), (x + w - i, y + h - i), (255 - i * 50, 255 - i * 30, 0), 2)

# Detect faces in the frame and draw bounding boxes
def detect_faces(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a gradient bounding box around the face
        draw_gradient_box(frame, x, y, w, h)

        # Add the label for face
        draw_label(frame, "Face", (x, y), label_color)

    return frame

# Apply a sharpening filter for better clarity in blurry images
def sharpen_image(frame):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening kernel
    sharpened = cv.filter2D(frame, -1, kernel)
    return sharpened

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the image (sharpening for blurry frames)
    frame = sharpen_image(frame)

    # Detect faces in the sharpened frame
    frame = detect_faces(frame)

    # Display the processed frame
    cv.imshow('Face Detection', frame)

    frame_small = cv.resize(frame, (420, 340))


    # Press 'q' to exit
    if cv.waitKey(1) == ord('q'):
        break

# Release resources and close windows
cap.release()
cv.destroyAllWindows()
