# Step 1: Import Required Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split

# Step 2: Data Preprocessing

# Define the path to your dataset folder
dataset_path = 'path_to_your_dataset'

# Image Preprocessing function (resize images and normalize them)
def preprocess_image(image_path, size=(100, 100)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize the image
    return img

# Load images and labels from your dataset
categories = ['with_mask', 'without_mask', 'incorrect_mask']
data = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Label encoding: with_mask=0, without_mask=1, incorrect_mask=2
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        img = preprocess_image(image_path)
        data.append(img)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Step 4: Data Augmentation (Optional, to improve model performance)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Step 5: Define the CNN Model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the data to feed into fully connected layers
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (3 categories: with_mask, without_mask, incorrect_mask)
model.add(Dense(3, activation='softmax'))

# Step 6: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Step 8: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Step 9: Visualize the Training Process (Accuracy & Loss)
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Step 10: Making Predictions
# Load a test image and predict its class
def predict_mask(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    label = np.argmax(prediction)
    categories = ['with_mask', 'without_mask', 'incorrect_mask']
    return categories[label]

# Test with an example image (replace with actual path)
image_path = 'path_to_test_image.jpg'
result = predict_mask(image_path)
print(f"Prediction: {result}")
