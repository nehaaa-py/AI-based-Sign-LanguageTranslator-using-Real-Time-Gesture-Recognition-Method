import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Load trained model and labels
model = tf.keras.models.load_model("asl_model.h5")
label_classes = np.load("label_classes.npy")

# Setup text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Start webcam
cap = cv2.VideoCapture(0)

prev_label = ""
stable_label = ""
same_count = 0

print("Press 'q' to quit, 's' to speak the predicted label.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and crop region of interest (ROI)
    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    input_image = normalized.reshape(1, 64, 64, 1)

    # Predict
    prediction = model.predict(input_image, verbose=0)
    current_label = label_classes[np.argmax(prediction)]

    # Stabilize predictions
    if current_label == prev_label:
        same_count += 1
    else:
        same_count = 0
        prev_label = current_label

    if same_count >= 5:
        stable_label = current_label
    else:
        stable_label = ""

    # Draw UI
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Detected: {stable_label}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and stable_label != "":
        print("Speaking:", stable_label)
        engine.say(stable_label)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
