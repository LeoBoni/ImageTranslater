import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/libras_alphabet_model.h5')
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_normalized = np.expand_dims(roi_normalized, axis=0)

    pred = model.predict(roi_normalized)
    letter = labels[np.argmax(pred)]

    cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.imshow('Alfabeto em Libras', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
