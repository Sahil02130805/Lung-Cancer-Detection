import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load images from directory
def load_images(folder, img_size=(224, 224)):
    images, labels = [], []
    for label in os.listdir(folder):  # 'cancerous', 'non_cancerous'
        label_path = os.path.join(folder, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)  # Resize to 224x224
            image = image / 255.0  # Normalize
            images.append(image)
            labels.append(1 if label == "cancerous" else 0)  
    return np.array(images), np.array(labels)

# Augment images
def augment_data(X, y):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, brightness_range=[0.8, 1.2], horizontal_flip=True)
    return datagen.flow(X, y, batch_size=32)

# Load dataset
def prepare_dataset():
    X, y = load_images("../data")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_gen = augment_data(X_train, y_train)
    return train_gen, (X_val, y_val)
