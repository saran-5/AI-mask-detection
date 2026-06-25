import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="sgd",
    loss="mean_squared_error"
)

model.fit(x, y, epochs=100, verbose=0)

prediction = model.predict(np.array([[7.0]]), verbose=0)

print("Prediction for 7 =", prediction[0][0])