import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from Outsourcing import DataPreprocessing


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

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
logdir = logfile_path + "\\graph\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Parameter
batch_size = 30
epochs = 200

# column order in CSV file
# Labels
# Iris setosa = 0
# Iris versicolor = 1
# Iris virginica = 2
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

features, labels = next(iter(train_dataset))

print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# Make Predictions
predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# Define the loss and gradient function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))

# Training Loop
# Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

# Paths for Tensorboard
current_time = now.strftime("%Y%m%d-%H%M%S")
train_log_dir = logfile_path + '\\train_logs'
test_log_dir = logfile_path + '\\test_logs'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


for epoch in range(epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop
  for x, y in dataset_setosa:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())


  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()


for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

  with test_summary_writer.as_default():
      tf.summary.scalar('accuracy', test_accuracy.result(), step=1)


print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

tf.stack([y, prediction], axis=1)

print("pred per class")
print(tf.nn.softmax(predictions[:5]))

# Prediction using the Model with unlabeled examples

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

predict_dataset_sorted = tf.convert_to_tensor([
    [5.0, 2.3, 3.3, 1.0, ],  # versicolor
    [7.0, 3.2, 4.7, 1.4, ],  # versicolor
    [5.0, 2.4, 3.8, 1.1, ],  # versicolor
    [7.7, 3.8, 6.7, 2.2, ],  # virginica
    [5.8, 2.8, 5.1, 2.4, ],  # virginica
    [6.3, 2.5, 5.0, 1.9]     # virginica
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset_sorted, training=False)

for i, logits in enumerate(predict_dataset_sorted):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
