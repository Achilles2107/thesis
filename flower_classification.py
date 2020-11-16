from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_federated as tff
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score
import nest_asyncio
nest_asyncio.apply()

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# train.reindex(np.random.permutation(train.index))

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe


print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()  # the species column is now gone

print(train.shape)  # we have 120 entires with 4 features)


initial_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

initial_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Display the model's architecture
initial_model.summary()

history = initial_model.fit(train, train_y, epochs=10, validation_data=(test, test_y),)  # we pass the data, labels and epochs and watch the magic!


initial_model.save('saved_model/iris_classification')

results = initial_model.evaluate(test, test_y, verbose=1)
print("test loss, test acc:", results)

# Federated Learning
df = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
df.reindex(np.random.permutation(df.index))

# Pick a subset of client devices to participate in training.
train_data = df

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
  initial_model,
  client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.1))
state = trainer.initialize()
for _ in range(5):
  state, metrics = trainer.next(state, train_data)
  print (metrics)





