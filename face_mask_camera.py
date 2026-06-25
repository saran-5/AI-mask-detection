"""
face_mask_camera.py — Real-Time Face Mask Detection via Webcam
==============================================================
Loads the trained Keras model, opens the default webcam, detects faces with
Haar Cascade, and classifies each face as MASK / NO MASK in real time.

Prerequisites:
    • mask_detector.keras  (produced by train.py)
    • haarcascade_frontalface_default.xml  (bundled with OpenCV — auto-located)

Usage:
    python face_mask_camera.py

Controls:
    q  →  quit
"""

import sys
import os
import cv2
import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "mask_detector.keras"
IMG_SIZE     = (128, 128)        # Must match what the model was trained on
CAMERA_INDEX = 0                 # 0 = default webcam; try 1 or 2 for external

# Labels must match the alphabetical order TensorFlow assigned during training:
#   'with_mask'    → class index 0
#   'without_mask' → class index 1
CLASS_NAMES  = ["with_mask", "without_mask"]

# Visual styling ──────────────────────────────────────────────────────────────
COLOR_MASK      = (0, 200, 0)     # BGR green  — face with mask
COLOR_NO_MASK   = (0, 0, 220)     # BGR red    — face without mask
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.8
FONT_THICKNESS  = 2
BOX_THICKNESS   = 2

# Haar Cascade detection parameters ───────────────────────────────────────────
SCALE_FACTOR    = 1.1            # How much image size is reduced at each scale
MIN_NEIGHBORS   = 5              # How many neighbours each candidate must have
MIN_FACE_SIZE   = (60, 60)       # Ignore tiny detections (noise)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the saved Keras model.
    Exits with a clear message if the file is missing — common when the user
    runs camera detection before running train.py.
    """
    if not os.path.exists(model_path):
        sys.exit(
            f"[ERROR] Model file '{model_path}' not found.\n"
            "Run 'python train.py' first to train and save the model."
        )
    print(f"[INFO] Loading model from '{model_path}' …")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD HAAR CASCADE
#    OpenCV ships the XML file alongside the cv2 package; we locate it
#    programmatically so the script works regardless of install path or OS.
# ─────────────────────────────────────────────────────────────────────────────

def load_face_cascade() -> cv2.CascadeClassifier:
    """
    Find and load the frontal-face Haar Cascade bundled with OpenCV.
    Falls back to looking in the current working directory if the bundled
    version cannot be located.
    """
    # Primary: use OpenCV's own data directory
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    if not os.path.exists(cascade_path):
        # Fallback: check current directory (e.g. user downloaded it manually)
        cascade_path = "haarcascade_frontalface_default.xml"

    if not os.path.exists(cascade_path):
        sys.exit(
            "[ERROR] haarcascade_frontalface_default.xml not found.\n"
            "It should be bundled with OpenCV. Re-installing OpenCV usually "
            "fixes this:\n    pip install --upgrade opencv-python"
        )

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        sys.exit("[ERROR] Failed to load Haar Cascade XML — file may be corrupt.")
    print(f"[INFO] Haar Cascade loaded from:\n       {cascade_path}")
    return cascade


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
#    Converts a single face region (NumPy BGR array) into the tensor format
#    expected by the model:
#      BGR → RGB
#      Resize to 128×128
#      Add batch dimension: shape (128,128,3) → (1,128,128,3)
#    The model's Rescaling layer handles pixel normalisation internally, so we
#    do NOT divide by 255 here.
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Return a (1, 128, 128, 3) float32 tensor ready for model.predict()."""
    face_rgb   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)                # (128, 128, 3)
    face_batch   = np.expand_dims(face_resized, axis=0)          # (1, 128, 128, 3)
    return face_batch.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION
#    The model outputs raw logits → softmax gives class probabilities.
#    We report the winning class and its confidence percentage.
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: tf.keras.Model, face_tensor: np.ndarray):
    """
    Returns:
        label      (str)   — "MASK" or "NO MASK"
        confidence (float) — 0.0 – 100.0
        color      (tuple) — BGR colour for the bounding box
    """
    logits      = model.predict(face_tensor, verbose=0)          # shape (1, 2)
    probs       = tf.nn.softmax(logits[0]).numpy()               # shape (2,)
    class_idx   = int(np.argmax(probs))
    confidence  = float(probs[class_idx]) * 100.0

    if CLASS_NAMES[class_idx] == "with_mask":
        label = "MASK"
        color = COLOR_MASK
    else:
        label = "NO MASK"
        color = COLOR_NO_MASK

    return label, confidence, color


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANNOTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_results(frame: np.ndarray, x: int, y: int, w: int, h: int,
                 label: str, confidence: float, color: tuple) -> None:
    """
    Draw a coloured bounding box and a label + confidence overlay on *frame*.
    The label is placed above the box; a filled rectangle behind the text
    ensures readability on any background.
    """
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

    # Text content, e.g. "MASK 98.5%"
    text = f"{label}  {confidence:.1f}%"

    # Measure text so we can size the background pill correctly
    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Background rectangle for the label (sits just above the bounding box)
    label_y = max(y - text_h - baseline - 6, 0)  # clamp to frame top
    cv2.rectangle(
        frame,
        (x, label_y),
        (x + text_w + 6, label_y + text_h + baseline + 6),
        color,
        thickness=cv2.FILLED,
    )

    # White text on the coloured background
    cv2.putText(
        frame,
        text,
        (x + 3, label_y + text_h + 3),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Display frames-per-second in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.putText(frame, text, (w - tw - 10, th + 10),
                FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(model: tf.keras.Model,
                  face_cascade: cv2.CascadeClassifier) -> None:
    """
    Open the webcam and run face-mask detection on every frame.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Cannot open camera (index {CAMERA_INDEX}).\n"
            "Ensure a webcam is connected. For an external camera try "
            "CAMERA_INDEX = 1 or 2."
        )

    print("[INFO] Webcam opened. Press 'q' in the preview window to quit.")

    # FPS calculation
    tick_freq   = cv2.getTickFrequency()
    prev_tick   = cv2.getTickCount()
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame received — skipping.")
            continue

        # ── FPS update ────────────────────────────────────────────────────
        curr_tick = cv2.getTickCount()
        fps = tick_freq / (curr_tick - prev_tick)
        prev_tick = curr_tick

        # ── Convert to grayscale for Haar Cascade ─────────────────────────
        #    The cascade detector works on intensity, not colour.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect faces ──────────────────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )
        # faces is a list of (x, y, w, h) rectangles, or an empty tuple

        # ── Classify each detected face ───────────────────────────────────
        for (x, y, w, h) in faces:
            # Crop the face region from the colour frame
            face_crop = frame[y : y + h, x : x + w]

            # Guard against degenerate crops (shouldn't happen, but be safe)
            if face_crop.size == 0:
                continue

            # Preprocess and predict
            face_tensor          = preprocess_face(face_crop)
            label, confidence, color = predict(model, face_tensor)

            # Draw bounding box + label
            draw_results(frame, x, y, w, h, label, confidence, color)

        # ── HUD ──────────────────────────────────────────────────────────
        draw_fps(frame, fps)
        face_count = len(faces) if isinstance(faces, np.ndarray) else 0
        cv2.putText(frame, f"Faces: {face_count}", (10, 28),
                    FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # ── Show frame ────────────────────────────────────────────────────
        cv2.imshow("Face Mask Detection  [press q to quit]", frame)

        # ── Quit on 'q' ───────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed — stopping.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model        = load_model(MODEL_PATH)
    face_cascade = load_face_cascade()
    run_detection(model, face_cascade)
