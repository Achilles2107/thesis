from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_federated as tff
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import h5py
import nest_asyncio
nest_asyncio.apply()

np.random.seed(0)

CSV_COLUMN_NAMES = [
        'psu_voltage', 'shunt_voltage',	'load_voltage',	'current',	'label',	'(MA)psu_voltage',	'(MA)shunt_voltage',	'(MA)load_voltage',	'(MA)current'
]
LABEL = ['0', '1', '2', '3', '4']

train_path = '/datasets/fan_lermer/20201113-131356/combined_csv.csv'
test_path = '/datasets/fan_lermer/20201113-131523/combined_csv.csv'

# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# train.reindex(np.random.permutation(train.index))

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('label')
test_y = test.pop('label')
train.head()  # the label column is now gone

dataset = tf.data.Dataset.from_tensor_slices((train.values, train_y.values))
print(dataset)

train_dataset = dataset.shuffle(len(train)).batch(1000)
# train_dataset = dataset.batch(100)

initial_model = tf.keras.Sequential([
  tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(8,), kernel_initializer='normal'),  # input shape required
  tf.keras.layers.Dense(30, activation=tf.nn.relu, kernel_initializer='normal'),
  tf.keras.layers.Dense(5)
])

initial_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Display the model's architecture
initial_model.summary()

training_history = initial_model.fit(train_dataset, epochs=20, validation_data=(test, test_y),)


results = initial_model.evaluate(test, test_y, verbose=1)
print("test loss, test acc:", results)

print(training_history.history.keys())

# summarize history for accuracy
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




