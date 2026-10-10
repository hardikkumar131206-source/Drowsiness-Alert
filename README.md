# Driver Drowsiness Detection System

A real-time drowsiness detector built with Python, OpenCV and a Keras CNN. It watches the driver's eyes through a webcam and sounds an alarm when they stay closed for too long.

> **Status:** educational project. It is a demonstration of a computer vision pipeline, not a certified safety product. See [Limitations](#limitations).

[Demo GIF or screenshot goes here]

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Approach](#the-approach)
3. [How It Works](#how-it-works)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Dataset](#dataset)
7. [Model](#model)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Results](#results)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Troubleshooting](#troubleshooting)
15. [Acknowledgements](#acknowledgements)
16. [License](#license)

---

## The Problem

Drowsy driving is a quiet cause of road accidents. Fatigue builds gradually: blinks get slower, eyes stay shut a little longer, and eventually a microsleep begins. The driver rarely notices the moment it happens, and by the time they do, it is often too late.

A system that can monitor the eyes continuously and react to a sustained closure can give the driver a timely warning.

## The Approach

The system answers one question on every video frame: **are the driver's eyes open or closed, and for how long?**

- A camera supplies the live video.
- Classical detection locates the face and eyes.
- A small convolutional neural network classifies each eye as open or closed.
- A real-time timer measures how long the eyes have stayed closed.
- If the closure lasts longer than a threshold (1.5 seconds by default), an alarm sounds and the screen flashes a red border.

A normal blink lasts a fraction of a second, so it never reaches the threshold and does not trigger the alarm.

## How It Works

```
Webcam frame
     |
     v
Convert to grayscale
     |
     v
Detect face (Haar cascade) -> use the largest face
     |
     v
Search for eyes only in the upper 60% of the face
     |
     v
Crop each eye -> resize to 32x32 grayscale
     |
     v
CNN classifies each eye: Open or Closed
     |
     v
Decision logic:
   any eye open         -> reset the timer
   all detected closed  -> start / continue the timer
   eyes not found       -> keep the timer unchanged
   no face              -> reset the timer
     |
     v
Timer >= threshold?  -> alarm + red border
```

### Design decisions

**Time-based logic instead of frame counting.** A frame counter makes behaviour depend on camera speed: 15 frames is about half a second at 30 FPS but about three seconds at 5 FPS. This project measures real elapsed time, so the threshold means the same thing on any hardware.

**Eyes are searched inside the face region only.** Searching the whole frame produces false detections such as nostrils, background objects and other faces. Restricting the search to the upper part of the largest detected face removes most of them.

**Normalisation lives inside the model.** The saved model rescales pixel values itself, so the detection script passes raw 0-255 pixels. This removes a common source of train/inference mismatch.

**Augmentation is applied to the data, not stored in the model.** The saved model contains only plain layers, which keeps it portable and saves cleanly to `.h5`.

**The alarm plays once and loops.** It does not stack overlapping sounds, and it stops as soon as the eyes open.

## Project Structure

```
drowsiness_project/
|-- haar cascade files/
|   |-- haarcascade_frontalface_alt.xml
|   |-- haarcascade_lefteye_2splits.xml
|   `-- haarcascade_righteye_2splits.xml
|-- models/
|   |-- cnnCat2.h5            (created by model.py)
|   `-- cnnCat2.keras         (created by model.py)
|-- dataset/                  (you add this, see Dataset)
|   |-- train/
|   |   |-- Open/
|   |   `-- Closed/
|   `-- test/
|       |-- Open/
|       `-- Closed/
|-- alarm.wav
|-- drowsiness detection.py   (main real-time application)
|-- model.py                  (training script)
|-- requirements.txt
`-- README.md
```

| File | Purpose |
|------|---------|
| `model.py` | Trains the CNN on the dataset and saves it to `models/` |
| `drowsiness detection.py` | Runs the live detector using the webcam |
| `haar cascade files/` | Pre-trained OpenCV cascades for face and eye detection |
| `alarm.wav` | Alert sound (a simple beep is generated automatically if this file is missing) |

## Tech Stack

| Component | Library |
|-----------|---------|
| Video capture and image processing | OpenCV (`opencv-python`, version 4.x) |
| Face and eye detection | Haar cascade classifiers |
| Eye state classification | TensorFlow / Keras CNN |
| Alarm playback | Pygame |
| Numerical operations | NumPy |

## Dataset

The model is trained on the **Yawn Eye Dataset New** from Kaggle:
<https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new>

The dataset contains four classes: `Open`, `Closed`, `yawn` and `no_yawn`. This project uses only the **`Open`** and **`Closed`** eye images. The yawn folders are ignored automatically by the training script.

| Split | Open | Closed |
|-------|------|--------|
| Train | 617 | 617 |
| Test | 109 | 109 |

Download the dataset, extract it, and place the `train` and `test` folders inside a `dataset` folder in the project root. Folder names must be exactly `Open` and `Closed`. Please check the dataset page for its licence and terms before redistributing any of its images.

## Model

A compact CNN that takes a **32x32 grayscale** eye image and outputs two class probabilities (`Closed` = 0, `Open` = 1).

| Layer | Details |
|-------|---------|
| Input | 32 x 32 x 1 |
| Rescaling | Divides pixels by 255 |
| Conv2D | 32 filters, 3x3, ReLU, same padding |
| BatchNormalization | |
| Conv2D | 32 filters, 3x3, ReLU, same padding |
| MaxPooling2D | 2x2 |
| Conv2D | 64 filters, 3x3, ReLU, same padding |
| BatchNormalization | |
| Conv2D | 64 filters, 3x3, ReLU, same padding |
| MaxPooling2D | 2x2 |
| Flatten | |
| Dropout | 0.4 |
| Dense | 128 units, ReLU |
| Dense (output) | 2 units, Softmax |

**Training setup**

- Optimiser: Adam (learning rate 1e-3)
- Loss: categorical cross-entropy
- Batch size: 32, up to 40 epochs
- Early stopping on validation accuracy (patience 10, best weights restored)
- Learning rate reduced on plateau
- Augmentation: horizontal flip, small rotation, zoom, contrast and brightness changes

## Installation

Tested on Windows 10 with Python 3.12.

**1. Clone the repository**

```
git clone [your-repository-url]
cd [your-repository-folder]
```

**2. Create and activate a virtual environment**

```
py -3.12 -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```
python3.12 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies**

```
pip install -r requirements.txt
```

or directly:

```
pip install tensorflow "opencv-python<5" pygame numpy
```

> **Important:** install OpenCV **4.x**. OpenCV 5.0 does not expose `cv2.CascadeClassifier`, which this project relies on, and fails with `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.

> **Python version:** TensorFlow support for very new Python releases can lag behind. Python 3.11 or 3.12 is the safest choice.

## Usage

**Step 1: add the dataset** as described in [Dataset](#dataset).

**Step 2: train the model** (once)

```
python model.py
```

This takes roughly 5 to 15 minutes on a normal laptop CPU. It prints the test accuracy, the per-class accuracy, and saves the model to `models/cnnCat2.h5` and `models/cnnCat2.keras`.

**Step 3: run the detector**

```
python "drowsiness detection.py"
```

The webcam window opens. Press **`q`** (with the video window focused) to quit.

### On-screen indicators

| Display | Meaning |
|---------|---------|
| `Open` | At least one detected eye is open |
| `Closed` | All detected eyes are closed |
| `Eyes not found` | A face is visible but no eyes were detected; the timer is held |
| `No face` | No face in view; the timer is reset |
| `Closed: x.xs` | How long the eyes have been closed continuously |
| Red border and `WAKE UP!` | The closure threshold was exceeded and the alarm is playing |

You only need to run `model.py` again if you want to retrain, for example after adding your own images.

## Configuration

| Setting | File | Default | Effect |
|---------|------|---------|--------|
| `CLOSED_SECONDS` | `drowsiness detection.py` | `1.5` | Seconds of continuous closure before the alarm sounds |
| `IMG` | `drowsiness detection.py` and `model.py` | `32` | Eye image size. **Must be identical in both files** |
| `EPOCHS` | `model.py` | `40` | Maximum training epochs |
| `BATCH` | `model.py` | `32` | Training batch size |

If you change `IMG`, retrain the model before running the detector.

## Results

On the held-out test split (218 eye images the model never trained on), the model reached **100% accuracy** for both the `Open` and `Closed` classes.

**How to read this number honestly:**

- The test set is small and comes from the same source and style as the training data (tightly cropped eyes).
- The test split was also used to choose when to stop training, so the figure is somewhat optimistic.
- Live performance depends on the Haar cascade crops, the camera, lighting and the person, and is expected to be lower than the test score.

Replace this section with your own live-webcam observations, for example how many deliberate long closures were caught out of how many attempts, to give a more realistic picture.

## Limitations

- **Lighting.** Standard webcams and Haar cascades perform poorly in low light. Real in-vehicle systems use infrared cameras.
- **Glasses and head pose.** Haar cascades are weaker with glasses, strong reflections and turned heads.
- **Looking down.** A driver looking at a phone has closed-looking eyes and is not distinguished from a driver whose eyes are shut.
- **Late indicator.** Prolonged eye closure signals drowsiness that has already begun. Earlier cues such as yawning and head nodding are not used.
- **Small training set.** About 1,200 training images cannot cover the diversity of faces, skin tones, ages, eye shapes and cameras found in real use.
- **Single driver, single face.** Only the largest face in view is analysed.
- **Privacy and safety.** The system is a learning project and must not be relied on to keep a driver safe.

## Future Work

- Replace Haar cascades with facial landmarks (for example MediaPipe Face Mesh) and compute the **Eye Aspect Ratio (EAR)** for more robust eye-state detection.
- Add **PERCLOS** (percentage of eye closure over a time window) as a more reliable fatigue measure.
- Add **yawn detection** and **head-pose estimation** for earlier warnings.
- Collect and train on images from the target webcam and a wider range of people and lighting.
- Evaluate with a proper confusion matrix, precision and recall on a larger independent test set.
- Support infrared cameras and run on embedded hardware such as a Raspberry Pi or Jetson.

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| `module 'cv2' has no attribute 'CascadeClassifier'` | OpenCV 5.x is installed. Run `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`, then `pip install "opencv-python<5"` |
| TensorFlow fails to install | Use Python 3.11 or 3.12 |
| `Model not found in models folder` | Run `python model.py` first |
| `Camera could not be opened` | Close other apps using the camera (Zoom, Teams, browser tabs) |
| `Eyes not found` appears often | Improve lighting, face the camera directly, remove glasses if possible |
| Open eyes reported as closed | Retrain with images from your own webcam added to `dataset/train` |
| Training error about missing folders | Check that `dataset/train/Open`, `dataset/train/Closed`, `dataset/test/Open` and `dataset/test/Closed` exist |
| `q` does not quit | Click the video window first so it has keyboard focus, or press `Ctrl + C` in the terminal |

## Acknowledgements

- The concept follows the widely shared Python drowsiness detection tutorial that combines Haar cascades with a CNN eye classifier. This implementation updates the code for current TensorFlow and OpenCV, replaces frame counting with time-based logic, restricts eye search to the face region, and retrains the model on the Kaggle dataset above.
- Haar cascade files are distributed with OpenCV.
- Dataset: *Yawn Eye Dataset New* on Kaggle.


## Author

[Hardik Kumar] - [https://www.linkedin.com/in/hardik-kumar-7631a832b/]
