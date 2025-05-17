"""
main.py - Plant Disease Classification using CNN
Author: HuongPham
Date: 2025-05-17
Description: Trains and evaluates a CNN for plant disease classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report


# === Constants ===
TRAIN_DIR = "/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/train"
VALID_DIR = "/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/valid"
MODEL_PATH = "trained_model_v6.keras"
HIST_PATH = "training_hist_v6.json"
ACC_PLOT_PATH = "accuracy_plot_v6.png"
CM_PLOT_PATH = "confusion_matrix_v6.png"
F1_SCORE_PLOT_PATH = "F1_score_plot_v6"
LOSS_PLOT_PATH = "Loss_poth_v6"
LR_PLOT_PATH = "LR_poth_v6"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 38
NUM_EPOCHS = 15

# === Utility Functions ===
def get_dataset(directory, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=shuffle
    )

def build_cnn(input_shape=(128, 128, 3), num_classes=38):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(32, kernel_size=3))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Block 2
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(64, kernel_size=3))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Block 3
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(128, kernel_size=3))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Block 4
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Block 5
    model.add(Conv2D(512, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(512, kernel_size=3))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Fully connected
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1500, activation='relu')) 
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def evaluate_and_print(model, dataset, name):
    loss, acc = model.evaluate(dataset)
    print(f"{name} accuracy: {acc:.6f}")
    return loss, acc

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# === Data Preparation ===
training_set = get_dataset(TRAIN_DIR)
validation_set = get_dataset(VALID_DIR)

# === Model Definition ===
cnn = build_cnn(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
cnn.summary()

# === Compile Model ===
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5)
]

# === Train Model ===
training_history = cnn.fit(
    training_set,
    validation_data=validation_set,
    epochs=NUM_EPOCHS,
    callbacks=callbacks
)

# === Evaluate Model ===
train_loss, train_acc = evaluate_and_print(cnn, training_set, "Training")
val_loss, val_acc = evaluate_and_print(cnn, validation_set, "Validation")

# === Save Model and History ===
cnn.save(MODEL_PATH)
with open(HIST_PATH, 'w') as f:
    json.dump(convert_to_serializable(training_history.history), f)

# === Plot Accuracy ===
plt.figure()
epochs = range(1, len(training_history.history['accuracy']) + 1)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(ACC_PLOT_PATH)
plt.close()

# === Plot Loss ===
plt.figure()
plt.plot(epochs, training_history.history['loss'], color='orange', label='Training Loss')
plt.plot(epochs, training_history.history['val_loss'], color='green', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.close()

# === Confusion Matrix ===
test_set = get_dataset(VALID_DIR, batch_size=1, shuffle=False)
y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1).numpy()
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1).numpy()
cm = confusion_matrix(Y_true, predicted_categories)

plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=True, annot_kws={"size": 10}, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Actual Class', fontsize=20)
plt.title('Plant Disease Prediction Confusion Matrix', fontsize=25)
plt.tight_layout()
plt.savefig(CM_PLOT_PATH)
plt.close()


# === Precision, Recall, F1-Score per class ===
CLASS_NAMES = test_set.class_names 
report = classification_report(Y_true, predicted_categories, target_names=CLASS_NAMES, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(12, 6))
sns.barplot(x=report_df.index[:-3], y=report_df['f1-score'][:-3])
plt.xticks(rotation=90)
plt.title('F1-Score per Class')
plt.tight_layout()
plt.savefig(F1_SCORE_PLOT_PATH)
plt.close()


# === Learning Rate Schedule ===

plt.plot(epochs, training_history.history['lr'])
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')
plt.savefig(LR_PLOT_PATH)
plt.close()


