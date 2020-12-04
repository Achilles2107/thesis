import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from Preprocessing import PreprocessData

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'
saved_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'

# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
# Tensorboard
now = datetime.now()

# Define the Keras TensorBoard callback.
logdir = logfile_path + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size)


# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

# Parameter
batch_size = 32
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Creating test and train datasets
# PreprocessData constructor usage
# url, filename, label_name, batch_size, title, shuffle_value=True,  column_names=None)
# Create Traindata
train_data = PreprocessData.PreprocessData(train_dataset_url, 'iris_training.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
train_data.get_dataset_by_url()
train_data.create_train_dataset()
train_data.make_graph()
train_data.map_dataset()
train_dataset = train_data.dataset

# Create Test Dataset
test_data = PreprocessData.PreprocessData(test_url, 'iris_test.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow', False, column_names)

test_data.get_dataset_by_url()
test_data.create_train_dataset()
test_data.make_graph()
test_data.map_dataset()
test_dataset = test_data.dataset

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Display the model's architecture
model.summary()

model.load_weights(checkpoint_path)

training_history = model.fit(train_dataset, epochs=epochs, validation_data=(test_dataset),
                             callbacks=[cp_callback])

model.save(saved_model_path + "\\keras_model\\")

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_dataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
