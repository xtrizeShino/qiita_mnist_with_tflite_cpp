import tensorflow as tf
import keras

if __name__ == '__main__':
	# for tensorflow 1.12
	# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("conv_mnist.h5")
	# for tensorflow-nightly (1.14)
    model = keras.models.load_model("conv_mnist.h5")
    print("Model loaded successfully")
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("conv_mnist.tflite", "wb").write(tflite_model)