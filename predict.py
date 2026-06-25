import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mask_detector.keras")

# Class names
class_names = ["with_mask", "without_mask"]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize for model
        face = cv2.resize(face, (128, 128))

        # Convert to array
        face = np.array(face, dtype=np.float32)

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)

        predicted_class = np.argmax(prediction)

        label = class_names[predicted_class]

        if label == "with_mask":
            color = (0, 255, 0)
            text = "MASK"
        else:
            color = (0, 0, 255)
            text = "NO MASK"

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Display result
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()