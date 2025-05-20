from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_dir = "/data/huongpham4/tmp_source/Dataset/PlantVillage"

import os
import shutil
import random

def split_dataset(full_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0

    for class_name in os.listdir(full_dir):
        class_path = os.path.join(full_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        sets = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }

        for set_name, image_list in sets.items():
            dest_dir = os.path.join(output_dir, set_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img in image_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest_dir, img))

split_dataset(
    full_dir="/data/huongpham4/tmp_source/Dataset/PlantVillage",
    output_dir="/data/huongpham4/tmp_source/Dataset/Data"
)
