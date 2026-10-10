"""
drowsiness detection.py  -  fixed, modern version

Run:  python "drowsiness detection.py"     (press q to quit)
Needs: models/cnnCat2.h5  (created by model.py)
"""
import math
import os
import struct
import time
import wave

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "cnnCat2.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE, "models", "cnnCat2.keras")
IMG = 32   
ALARM_PATH = os.path.join(BASE, "alarm.wav")
CLOSED_SECONDS = 1.5   


def make_alarm(path):
    rate, dur, freq = 44100, 1.0, 1000
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(rate * dur))
    )
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


if not os.path.exists(ALARM_PATH):
    make_alarm(ALARM_PATH)

mixer.init()
sound = mixer.Sound(ALARM_PATH)

haar = os.path.join(BASE, "haar cascade files")
face_cascade = cv2.CascadeClassifier(os.path.join(haar, "haarcascade_frontalface_alt.xml"))
leye_cascade = cv2.CascadeClassifier(os.path.join(haar, "haarcascade_lefteye_2splits.xml"))
reye_cascade = cv2.CascadeClassifier(os.path.join(haar, "haarcascade_righteye_2splits.xml"))
if face_cascade.empty() or leye_cascade.empty() or reye_cascade.empty():
    raise SystemExit("Haar cascade XML files not found in 'haar cascade files' folder.")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found in models folder. Run model.py first to train it.")

model = load_model(MODEL_PATH)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def eye_is_open(eye_gray):
    eye = cv2.resize(eye_gray, (IMG, IMG)).astype("float32")
    eye = eye.reshape(1, IMG, IMG, 1)          
    probs = model(eye, training=False).numpy()[0]
    return int(np.argmax(probs)) == 1


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera could not be opened.")

closed_since = None   
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))
    status = "No face"
    results = []   
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        roi_gray = gray[y:y + int(h * 0.6), x:x + w]

        for cascade in (leye_cascade, reye_cascade):
            eyes = cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                            minNeighbors=4)
            if len(eyes) > 0:
                ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
                crop = roi_gray[ey:ey + eh, ex:ex + ew]
                results.append(eye_is_open(crop))
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (0, 255, 0), 1)

        if results:
            if any(results):               
                status = "Open"
                closed_since = None
            else:                          
                status = "Closed"
                if closed_since is None:
                    closed_since = time.time()
        else:
            status = "Eyes not found"      
    else:
        closed_since = None

    closed_for = (time.time() - closed_since) if closed_since else 0.0
    drowsy = closed_for >= CLOSED_SECONDS

    cv2.rectangle(frame, (0, height - 50), (330, height), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, status, (10, height - 20), font, 1,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Closed: {closed_for:.1f}s", (130, height - 20), font,
                1, (255, 255, 255), 1, cv2.LINE_AA)

    if drowsy:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
        cv2.putText(frame, "WAKE UP!", (width // 2 - 80, 50), font, 2,
                    (0, 0, 255), 2, cv2.LINE_AA)
        if not alarm_on:                   
            sound.play(-1)
            alarm_on = True
    elif alarm_on:
        sound.stop()
        alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
