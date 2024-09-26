import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from ProcessImage import load_fer_data, load_utkface_data

# 1. Set up paths for both datasets
fer_train_dir = 'fer2013/train'
fer_test_dir = 'fer2013/test'
utkface_train_dir = 'utkface-new/train'
utkface_test_dir = 'utkface-new/test'

# =========================== Emotion Detection (FER-2013) ===========================
# 2. Load and preprocess the FER-2013 dataset
X_train_fer, y_train_fer = load_fer_data(fer_train_dir)
X_test_fer, y_test_fer = load_fer_data(fer_test_dir)

# One-hot encoding the emotion labels (for FER dataset)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
y_train_fer = np.array([emotion_to_idx[label] for label in y_train_fer])
y_test_fer = np.array([emotion_to_idx[label] for label in y_test_fer])
y_train_fer = to_categorical(y_train_fer, num_classes=len(emotion_labels))
y_test_fer = to_categorical(y_test_fer, num_classes=len(emotion_labels))

# 3. Build a deeper CNN for emotion detection with batch normalization and regularization
def create_emotion_model(input_shape=(48, 48, 1), num_classes=len(emotion_labels)):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))  # Prevent overfitting

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Train the emotion detection model
emotion_model = create_emotion_model()
emotion_model.summary()

# Reshape FER images for model input (48x48x1)
X_train_fer = X_train_fer.reshape(X_train_fer.shape[0], 48, 48, 1)
X_test_fer = X_test_fer.reshape(X_test_fer.shape[0], 48, 48, 1)

# Data augmentation (for robustness)
datagen_fer = ImageDataGenerator(rotation_range=20, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_fer.fit(X_train_fer)

# Callbacks: Reduce learning rate on plateau, Early stopping to avoid overfitting
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
emotion_model.fit(datagen_fer.flow(X_train_fer, y_train_fer, batch_size=32),
                  validation_data=(X_test_fer, y_test_fer), epochs=50, callbacks=[reduce_lr, early_stop])

# Save the trained model
emotion_model.save('emotion_detection_model.h5')

# ============================= Age Detection (UTKFace) =============================
# 5. Load and preprocess the UTKFace dataset
X_train_utk, y_train_utk = load_utkface_data(utkface_train_dir)
X_test_utk, y_test_utk = load_utkface_data(utkface_test_dir)

# For age prediction, using classification, create age ranges
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
age_to_idx = {age: idx for idx, age in enumerate(age_labels)}
y_train_utk = np.array([age_to_idx[label] for label in y_train_utk])
y_test_utk = np.array([age_to_idx[label] for label in y_test_utk])
y_train_utk = to_categorical(y_train_utk, num_classes=len(age_labels))
y_test_utk = to_categorical(y_test_utk, num_classes=len(age_labels))

# 6. Build a deeper CNN for age detection
def create_age_model(input_shape=(64, 64, 3), num_classes=len(age_labels)):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 7. Train the age detection model
age_model = create_age_model()
age_model.summary()

# Reshape UTKFace images for model input (64x64x3)
X_train_utk = X_train_utk.reshape(X_train_utk.shape[0], 64, 64, 3)
X_test_utk = X_test_utk.reshape(X_test_utk.shape[0], 64, 64, 3)

# Data augmentation
datagen_utk = ImageDataGenerator(rotation_range=20, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_utk.fit(X_train_utk)

# Train the age model
age_model.fit(datagen_utk.flow(X_train_utk, y_train_utk, batch_size=32),
              validation_data=(X_test_utk, y_test_utk), epochs=50, callbacks=[reduce_lr, early_stop])

# Save the trained model
age_model.save('age_detection_model.h5')
