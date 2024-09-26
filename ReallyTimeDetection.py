import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained models
emotion_model = load_model('emotion_detection_model.h5', compile=False)
age_model = load_model('age_detection_model.h5', compile=False)

# Define emotion labels and age ranges
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

# Initialize OpenCV's face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale (for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Preprocess the face for emotion detection (convert to grayscale)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized_emotion = cv2.resize(face_gray, (48, 48))  # Resize for emotion model
        face_resized_emotion = face_resized_emotion / 255.0
        face_resized_emotion = np.expand_dims(face_resized_emotion, axis=-1)  # Add channel for grayscale
        face_resized_emotion = np.expand_dims(face_resized_emotion, axis=0)  # Add batch dimension

        # Preprocess the face for age detection (keep as RGB)
        face_resized_age = cv2.resize(face, (64, 64))  # Resize for age model
        face_resized_age = face_resized_age / 255.0
        face_resized_age = np.expand_dims(face_resized_age, axis=0)  # Add batch dimension

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_resized_emotion)
        emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[emotion_index]

        # Predict age
        age_prediction = age_model.predict(face_resized_age)
        age_index = np.argmax(age_prediction)
        predicted_age = age_labels[age_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the emotion and age on the video frame
        label = f'{predicted_emotion}, {predicted_age}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion and Age Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Check if the window is closed using the "X" button
    if cv2.getWindowProperty('Emotion and Age Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
