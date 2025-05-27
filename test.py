import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === CONFIG ===
TEST_DIR = "/data/huongpham4/tmp_source/Dataset/Data/test"
IMG_SIZE = (128, 128)
MODEL_PATH = "trained_model_v12.keras" 

# === LOAD MODEL ===
model = load_model(MODEL_PATH)

class_names = sorted(os.listdir(TEST_DIR))  

def predict_image(img_path):
    image = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(image).astype(np.float32) / 1.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    return predicted_class

for class_folder in os.listdir(TEST_DIR):
    folder_path = os.path.join(TEST_DIR, class_folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        predicted_class = predict_image(img_path)
        print(f"Image: {img_name:<30} | Actual: {class_folder:<40} | Predicted: {predicted_class}")
