import cv2
import os

cap = cv2.VideoCapture(0)
letter = 'A'  # Mude para a letra que você deseja capturar
save_path = f'dataset\\{letter}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = frame[100:300, 100:300]
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        img_name = f"{save_path}\\{count}.png"
        cv2.imwrite(img_name, roi)
        print(f"Saved: {img_name}")
        count += 1
    elif k == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
