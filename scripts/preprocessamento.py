from PIL import Image
import os
import numpy as np

def preprocess_image(image_path, output_size=(64, 64)):
    img = Image.open(image_path)
    img = img.resize(output_size)
    img = img.convert('RGB')
    return np.array(img) / 255.0

for letter in os.listdir('dataset'):
    folder_path = f'dataset\\{letter}'
    for img_file in os.listdir(folder_path):
        if img_file.endswith('.png'):
            img_path = os.path.join(folder_path, img_file)
            img = preprocess_image(img_path)
            np.save(img_path.replace('.png', '.npy'), img)
