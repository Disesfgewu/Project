import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
    print("GPU Device Name:", tf.config.list_physical_devices('GPU'))
else:
    print("GPU is not available.")