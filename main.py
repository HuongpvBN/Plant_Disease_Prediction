import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


train_dir = "/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/train"
valid_dir = "/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/valid"

training_set = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

cnn = Sequential()

# Block 1
cnn.add(tf.keras.Input(shape=(128, 128, 3)))  
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(Conv2D(32, kernel_size=3))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Block 2
cnn.add(Conv2D(64, kernel_size=3, padding='same'))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(Conv2D(64, kernel_size=3))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Block 3
cnn.add(Conv2D(128, kernel_size=3, padding='same'))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(Conv2D(128, kernel_size=3))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Block 4
cnn.add(Conv2D(256, kernel_size=3, padding='same'))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(Conv2D(256, kernel_size=3))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Block 5
cnn.add(Conv2D(512, kernel_size=3, padding='same'))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(Conv2D(512, kernel_size=3))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Fully connected
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(1500, activation='relu'))
cnn.add(Dropout(0.4))
cnn.add(Dense(38, activation='softmax'))

cnn.summary()


cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

training_history  = cnn.fit(
    training_set,
    validation_data=validation_set,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ])


#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)


cnn.save('trained_model.keras')

training_history.history
with open('training_hist.json','w') as f:
  json.dump(training_history.history, f)

print(training_history.history.keys())

epochs = [i for i in range(1, 11)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.savefig('accuracy_plot.png')  
plt.close()

class_name = validation_set.class_names

test_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False
)

y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1).numpy()  


true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1).numpy()  


cm = confusion_matrix(Y_true, predicted_categories)
print(classification_report(Y_true, predicted_categories, target_names=class_name))

plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=True, annot_kws={"size": 10}, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Actual Class', fontsize=20)
plt.title('Plant Disease Prediction Confusion Matrix', fontsize=25)
plt.savefig('confusion_matrix.png') 
plt.close()