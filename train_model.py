# train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data(folder):
    images, labels = [], []
    label_map = {}  # Map folder names to class indices
    class_names = sorted(os.listdir(folder))
    for idx, label_name in enumerate(class_names):
        label_map[label_name] = idx
        label_path = os.path.join(folder, label_name)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64)) / 255.0
            images.append(img)
            labels.append(idx)
    return (np.array(images).reshape(-1, 64, 64, 1),
            to_categorical(labels), class_names)

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

images, labels, class_names = load_data("dataset/asl_data_with_images")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = create_model(num_classes=len(class_names))
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
model.save("asl_model.h5")
np.save("label_classes.npy", np.array(class_names))


# Save class names
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
