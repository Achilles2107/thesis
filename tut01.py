#Code von https://www.youtube.com/watch?v=f1DYLlATaag&list=PLpXN7decvI0USQwHZwzxirlZiT6uUXrww&index=7&t=11s
#

import tensorflow as tf
import numpy as np
from tensorflow import keras


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError(),
                                                                    tf.keras.metrics.Accuracy])


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


model.fit(xs, ys, epochs=8000)

print(model.predict([10.0]))

