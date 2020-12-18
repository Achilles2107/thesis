import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
from Outsourcing import DataPreprocessing

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_train_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\train\\'
split_test_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\test\\'
saved_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'
# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

# Tensorboard
now = datetime.now()
# Define the Keras TensorBoard callback.
logdir = logfile_path + "\\graph\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Parameter
batch_size = 30
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create train and test data
# PreprocessData constructor usage
# url, filename, label_name, batch_size, title, shuffle_value=True,  column_names=None)
# Create Traindata
train_data = DataPreprocessing.PreprocessData(train_dataset_url, 'iris_training.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
train_data.get_dataset_by_url()
train_data.create_train_dataset()
train_data.make_graph()
train_data.map_dataset()
train_dataset = train_data.dataset

# Create Test Dataset
test_data = DataPreprocessing.PreprocessData(test_url, 'iris_test.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow', False, column_names)

test_data.get_dataset_by_url()
test_data.create_train_dataset()
test_data.make_graph()
test_data.map_dataset()
test_dataset = test_data.dataset

# Create split test data
split_test_data01 = DataPreprocessing.PreprocessData(split_test_data_path, '1.csv', label_name, batch_size,
                                                  'Split Test Dataset 01', True, column_names)
split_test_data01.get_local_dataset()
split_test_data01.create_train_dataset()
split_test_data01.make_graph()
split_test_data01.map_dataset()
split_testdataset01 = split_test_data01.dataset

split_test_data02 = DataPreprocessing.PreprocessData(split_test_data_path, '2.csv', label_name, batch_size,
                                                  'Split Test Dataset 02', True, column_names)
split_test_data02.get_local_dataset()
split_test_data02.create_train_dataset()
split_test_data02.make_graph()
split_test_data02.map_dataset()
split_testdataset02 = split_test_data02.dataset

split_test_data03 = DataPreprocessing.PreprocessData(split_test_data_path, '3.csv', label_name, batch_size,
                                                  'Split Test Dataset 03', True, column_names)
split_test_data03.get_local_dataset()
split_test_data03.create_train_dataset()
split_test_data03.make_graph()
split_test_data03.map_dataset()
split_testdataset03 = split_test_data01.dataset

# Create Split Dataset
split_data01 = DataPreprocessing.PreprocessData(split_train_data_path, '1.csv', label_name, batch_size,
                                                  'Split Dataset 01', True, column_names)
split_data01.get_local_dataset()
split_data01.create_train_dataset()
split_data01.make_graph()
split_data01.map_dataset()
split_dataset01 = split_data01.dataset

split_data02 = DataPreprocessing.PreprocessData(split_train_data_path, '2.csv', label_name, batch_size,
                                                  'Split Dataset 02', True, column_names)
split_data02.get_local_dataset()
split_data02.create_train_dataset()
split_data02.make_graph()
split_data02.map_dataset()
split_dataset02 = split_data02.dataset

split_data03 = DataPreprocessing.PreprocessData(split_train_data_path, '3.csv', label_name, batch_size,
                                                  'Split Dataset 03', True, column_names)
split_data03.get_local_dataset()
split_data03.create_train_dataset()
split_data03.make_graph()
split_data03.map_dataset()
split_dataset03 = split_data03.dataset


# Sorted Datasets
data_setosa = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_setosa.csv', label_name, batch_size,
                                                  'Iris setosa Dataset', True, column_names)
data_setosa.get_local_dataset()
data_setosa.create_train_dataset()
data_setosa.make_graph()
data_setosa.map_dataset()
dataset_setosa = data_setosa.dataset

data_versicolor = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_versicolor.csv', label_name, batch_size,
                                                  'Iris versicolor Dataset', True, column_names)
data_versicolor.get_local_dataset()
data_versicolor.create_train_dataset()
data_versicolor.make_graph()
data_versicolor.map_dataset()
dataset_versicolor = data_versicolor.dataset

data_virginica = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_versicolor.csv', label_name, batch_size,
                                                  'Iris versicolor Dataset', True, column_names)
data_virginica.get_local_dataset()
data_virginica.create_train_dataset()
data_virginica.make_graph()
data_virginica.map_dataset()
dataset_virginica = data_virginica.dataset

# Make lists with dataset per client
split_datasets = [split_dataset01, split_dataset02, split_dataset03]
test_datasets = [test_dataset, test_dataset, test_dataset]
sorted_datasets = [dataset_setosa, dataset_virginica, dataset_versicolor]

metrics_list = []


# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
  ])


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  # loading saved model from IrisModelSaveWeights.py
  keras_model = tf.keras.models.load_model(saved_model_path + "keras_model\\", compile=False)
  return tff.learning.from_keras_model(
      keras_model,
      # Input information on what shape the input data will have
      # Must be from type tf.Type or tf.TensorSpec
      input_spec=split_dataset01.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))  # for Global model

# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

tf.summary.trace_on(graph=True, profiler=True)
# Construct Server State
state = iterative_process.initialize()
with summary_writer.as_default():
  tf.summary.trace_export(
      name="Federated iterative process init",
      step=0,
      profiler_outdir=logfile_path)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

# Start Federated Learning process
for round_num in range(1, epochs):
  state, metrics = iterative_process.next(state, split_datasets)
  with summary_writer.as_default():
      for name, metric in metrics['train'].items():
          tf.summary.scalar(name, metric, step=round_num)
  train_metrics = metrics['train']
  print('round ' + str(round_num) + ' loss={l:.3f}, accuracy={a:.3f}'.format(
        l=train_metrics['loss'], a=train_metrics['sparse_categorical_accuracy']))

  train_loss_results.append(train_metrics['loss'])
  train_accuracy_results.append(train_metrics['sparse_categorical_accuracy'])

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# Print content of metrics['train']
# for name, metric in metrics['train'].items():
#     print(name, metric)

# Evaluation
# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, test_datasets)

print(str(train_metrics))
