import tensorflow as tf

model = tf.keras.models.load_model('trained_model_v12.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_v12.tflite', 'wb') as f:
    f.write(tflite_model)



