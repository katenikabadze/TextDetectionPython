import cv2
import pytesseract
from pytesseract import Output
import imutils
import numpy as np

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to make small text more visible
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Adaptive thresholding for better OCR
    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # Tesseract configuration: sparse + orientation detection
    config = "--oem 3 --psm 11"

    # Try OCR on original frame first
    d = pytesseract.image_to_data(gray, output_type=Output.DICT, config=config)

    # If no text detected, try rotated frames
    if not any(text.strip() != "" for text in d['text']):
        angles = [15, -15, 30, -30]
        for angle in angles:
            rotated = imutils.rotate_bound(gray, angle)
            d = pytesseract.image_to_data(rotated, output_type=Output.DICT, config=config)
            if any(text.strip() != "" for text in d['text']):
                break

    # Draw rectangles and text
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if d['conf'][i] != '-1' and int(d['conf'][i]) > 60:
            text = d['text'][i].strip()
            if text:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                y_text = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(frame, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the live frame
    cv2.imshow("Real-Time Text Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()