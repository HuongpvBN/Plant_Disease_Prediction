# import tensorflow as tf
# import numpy as np
# import os
# from tensorflow.keras.utils import image_dataset_from_directory
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Đường dẫn đến thư mục test ảnh
# test_dir = '/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/valid'

# # Load tập test như một dataset
# test_set = image_dataset_from_directory(
#     test_dir,
#     labels='inferred',
#     label_mode='categorical',  # dùng one-hot
#     image_size=(128, 128),
#     batch_size=1,
#     shuffle=False
# )

# class_names = test_set.class_names

# # Load TFLite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# # Lấy input/output tensor
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# y_true = []
# y_pred = []

# # Duyệt qua từng ảnh trong tập test
# for images, labels in test_set:
#     input_data = tf.cast(images, tf.float32).numpy()  # Đảm bảo đúng kiểu
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     predicted_label = np.argmax(output_data[0])
#     true_label = np.argmax(labels.numpy()[0])

#     y_pred.append(predicted_label)
#     y_true.append(true_label)

# # In báo cáo
# print("Classification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # Vẽ ma trận nhầm lẫn
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for TFLite Model')
# plt.tight_layout()
# plt.savefig("tflite_confusion_matrix.png")
# plt.close()


#==============================
# import time
# import numpy as np
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# valid_dir = '/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/valid'

# validation_set = tf.keras.utils.image_dataset_from_directory(
#     valid_dir,
#     labels="inferred",
#     label_mode="categorical",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False
# )
# class_name = validation_set.class_names
# print(class_name)

# cnn = tf.keras.models.load_model('trained_model.keras')


# import cv2
# image_path = '/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/test/test/AppleCedarRust1.JPG'
# # Reading an image in default mode
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
# # Displaying the image 
# # plt.imshow(img)
# # plt.title('Test Image')
# # plt.xticks([])
# # plt.yticks([])
# # plt.show()

# start_time = time.time()

# image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
# input_arr = tf.keras.preprocessing.image.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = cnn.predict(input_arr)
# end_time = time.time()
# elapsed_time = end_time - start_time

# result_index = np.argmax(predictions) #Return index of max element
# print(result_index)

# model_prediction = class_name[result_index]
# print(f"Disease Name: {model_prediction}")
# print(f"Thời gian xử lý: {elapsed_time:.8f} giây")


#==============================

# import time
# import numpy as np
# import tensorflow as tf
# import cv2

# # Load labels (class names)
# valid_dir = '/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/valid'
# validation_set = tf.keras.utils.image_dataset_from_directory(
#     valid_dir,
#     labels="inferred",
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(128, 128)
# )
# class_name = validation_set.class_names
# print(class_name)

# # Load TFLite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Load and preprocess image
# image_path = '/data/huongpham4/tmp_source/Plant_Disease_Dataset/Plant_Disease_Dataset/test/test/AppleCedarRust1.JPG'
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_resized = cv2.resize(img, (128, 128))
# input_arr = np.expand_dims(img_resized.astype(np.float32), axis=0)  # shape: (1, 128, 128, 3)

# # Quantize if needed
# if input_details[0]['dtype'] == np.uint8:
#     scale, zero_point = input_details[0]['quantization']
#     input_arr = (input_arr / 255.0 / scale + zero_point).astype(np.uint8)

# # --- Bắt đầu đo thời gian ---
# start_time = time.time()

# # Inference
# interpreter.set_tensor(input_details[0]['index'], input_arr)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])

# # --- Kết thúc đo thời gian ---
# end_time = time.time()
# elapsed_time = end_time - start_time

# # Hiển thị kết quả
# result_index = np.argmax(output_data)
# print(f"Disease Name: {class_name[result_index]}")
# print(f"Thời gian xử lý: {elapsed_time:.8f} giây")