# ⚽ Football Jersey Number Recognition Pipeline

### YOLO Tracking + Team Clustering + EasyOCR Voting

------------------------------------------------------------------------

## 📌 Overview

This project provides an **end-to-end football (soccer) video analytics
pipeline** capable of automatically:

-   Tracking players across frames
-   Separating teams based on jersey colors
-   Detecting and recognizing jersey numbers
-   Stabilizing OCR results over time
-   Producing an annotated output video

The system combines **object detection, multi-object tracking, computer
vision preprocessing, and OCR stabilization** to solve one of the
hardest broadcast-analysis problems:\
\> Reliable player jersey number recognition from real match footage.

------------------------------------------------------------------------

## 🎯 Key Features

-   **Persistent Player Tracking** using YOLO + BoT-SORT
-   **Automatic Team Classification** using KMeans on kit colors
-   **Grass Removal** to isolate jerseys
-   **Multi-ROI Jersey Extraction**
-   **Multi-Preprocessing OCR Pipeline**
-   **Voting & Locking System** to prevent flickering numbers
-   **Prefix-1 Recovery** (fixes common OCR error: `14 → 4`)
-   **Debug Mode** to inspect OCR failures
-   Annotated output video export

------------------------------------------------------------------------

## 🧠 Why This Is Hard

Jersey number recognition in football broadcasts is challenging because:

-   Players are small in frame
-   Motion blur occurs frequently
-   Jerseys are wrinkled and occluded
-   Lighting changes every frame
-   Digits like **1, 4, 7** are easily confused
-   Camera zoom constantly changes

This pipeline solves the problem by combining: \> tracking + temporal
voting + adaptive preprocessing instead of relying on single-frame OCR.

------------------------------------------------------------------------

## 🏗️ Pipeline Architecture

    Video
      ↓
    YOLO Detection
      ↓
    BoT-SORT Tracking (Persistent IDs)
      ↓
    Player Cropping
      ↓
    Grass Removal
      ↓
    KMeans Kit Clustering → Team Assignment
      ↓
    Multi-ROI Jersey Extraction
      ↓
    Image Enhancement (CLAHE + Thresholds)
      ↓
    EasyOCR (Digits Only)
      ↓
    Voting + Locking (Track Level)
      ↓
    Annotated Video Output

------------------------------------------------------------------------

## 📂 Project Structure

    project/
    │
    ├── notebook.ipynb
    ├── weights/
    │   └── best.pt
    ├── test_videos/
    │   └── CV_Task.mp4
    ├── output/
    │   └── *_out.mp4
    ├── debug_ocr/        # auto-generated crops
    └── README.md

------------------------------------------------------------------------

## ⚙️ Requirements

### Python

Python 3.9 -- 3.11 recommended

### Install Dependencies

``` bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install tqdm
pip install scikit-learn
pip install easyocr
pip install torch torchvision torchaudio
```

------------------------------------------------------------------------

## 🚀 How To Run

1️⃣ Place your trained YOLO model:

    weights/best.pt

2️⃣ Put your input video:

    test_videos/CV_Task.mp4

3️⃣ Open the notebook:

    Jupyter Notebook

4️⃣ Run all cells

The output video will appear in:

    /output/

------------------------------------------------------------------------

## 🧪 Debug Mode

The pipeline can automatically save OCR crops.

Enable in notebook:

``` python
DEBUG_OCR = True
```

Crops will be saved in:

    debug_ocr/

This is extremely useful for: - tuning OCR thresholds - analyzing
mis-detections - training future models

------------------------------------------------------------------------

## 🔍 OCR Stabilization Logic

The system does NOT trust single-frame OCR.

Instead it: 1. Reads numbers every frame 2. Stores them per tracking ID
3. Filters weak confidence reads 4. Calculates a stability score 5.
Locks the number only when stable

### Stability Score

    score = frequency × average_confidence

This prevents:

    8 → 11 → 7 → 8 → 3 → 8   (flickering)

and produces:

    #8 (locked)

------------------------------------------------------------------------

## 🛠️ Important Fixes Included

### Prefix-1 Recovery

Fixes common OCR failure:

  Actual   OCR   Fixed
  -------- ----- -------
  14       4     14
  17       7     17
  11       1     11

The algorithm checks a thin left strip of the jersey for digit "1".

------------------------------------------------------------------------

## 🧾 Output

The pipeline produces an annotated video:

-   Player bounding boxes
-   Team side labels
-   Stable jersey numbers

Example:

    Player-L   #8
    Player-R   #23
    GK-L       #1

------------------------------------------------------------------------

## 📈 Possible Extensions

-   Player re-identification (ReID embeddings)
-   Pass detection
-   Ball possession tracking
-   Automatic match statistics
-   Formation analysis
-   Dataset auto-labeling for training

------------------------------------------------------------------------

## ⚠️ Known Limitations

-   Very small players far from camera
-   Heavy occlusion (arms covering numbers)
-   Goalkeeper long-sleeve wrinkled jerseys
-   Extremely low resolution broadcasts

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome.\
If you improve OCR robustness or tracking stability, feel free to
submit.


------------------------------------------------------------------------

## 🙌 Acknowledgments

-   Ultralytics YOLO
-   EasyOCR
-   OpenCV
-   Scikit-Learn KMeans
