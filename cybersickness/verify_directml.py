import tensorflow as tf


print("TensorFlow:", tf.__version__)
print("All devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# Warm-up on GPU if available
with tf.device("/GPU:0"):
    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)
print("Matmul ok, shape:", c.shape)
