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
# Here we use keras (a module inside of TensorFlow) to grab our Datasets and read them into a pandas dataframe

print(type(train))
print(train.dtypes)
print(type(test))
print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()  # the species column is now gone


dataset = tf.data.Dataset.from_tensor_slices((train.values, train_y.values))
print(dataset)

for feat, spec in dataset.take(5):
  print ('Features: {}, Species: {}'.format(feat, spec))

train_dataset = dataset.shuffle(len(train)).batch(40)
# train_dataset = dataset.batch(120)

print(train.shape)  # we have 120 entires with 4 features)


initial_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,), kernel_initializer='normal'),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_initializer='normal'),
  tf.keras.layers.Dense(3)
])

initial_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Display the model's architecture
initial_model.summary()

# history = initial_model.fit(train, train_y, epochs=10, validation_data=(test, test_y),)

training_history = initial_model.fit(train_dataset, epochs=200, validation_data=(test, test_y),)

# initial_model.save('Storage/IrisClassification')

# tf.keras.models.save_model(initial_model, 'Storage/IrisModel', include_optimizer=False)

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


# IrisFederatedLearning
federated = False

if federated == True:

    df = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df['id'] = range(1, len(df) + 1)

    print(df.head())

    dataset = tf.data.Dataset.from_tensor_slices((train.values, train_y.values))
    print(dataset)

    client_id_colname = 'id'  # the column that represents client ID
    SHUFFLE_BUFFER = 40
    NUM_EPOCHS = 5
    NUM_CLIENTS = 5
    BATCH_SIZE = 40

    train_data = dataset

    # Wrap a Keras model for use with TFF.
    def model_fn():
      modelfn = tf.keras.models.Sequential([
          tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,), kernel_initializer='normal'),
          tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_initializer='normal'),
          tf.keras.layers.Dense(3)
      ])
      return tff.learning.from_keras_model(
          modelfn,
          input_spec=train_data,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
      model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.1))
    state = trainer.initialize()
    for _ in range(5):
      state, metrics = trainer.next(state, train_data)
      print(metrics)






