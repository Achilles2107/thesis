import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
save_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'


# Tensorboard
now = datetime.now()

# Define the Keras TensorBoard callback.
logdir = logfile_path + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Parameter
batch_size = 32
epochs = 200

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Labels
# Iris setosa = 0
# Iris versicolor = 1
# Iris virginica = 2

# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'


# For Datasets with Feature and Label Names in the actual File
def create_train_dataset(file_path, filename):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    #column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    return train_dataset


# For Datasets without Feature and Label Names in the actual File
def create_train_dataset_with_col_name(file_path, filename):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    return train_dataset


# CSV from Github
train_dataset_iris_github = tf.data.experimental.make_csv_dataset(
    github_dataset,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)


print("train_dataset Features")
features, labels = next(iter(train_dataset))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


# Datasets Sorted by Species
dataset_versicolor = create_train_dataset_with_col_name(dataset_path_local, 'iris_versicolor.csv')
dataset_setosa = create_train_dataset_with_col_name(dataset_path_local, 'iris_setosa.csv')
dataset_virginica = create_train_dataset_with_col_name(dataset_path_local, 'iris_virginica.csv')

# Pack Datasets
# Sorted Datasets
dataset_versicolor = dataset_versicolor.map(pack_features_vector)
dataset_virginica = dataset_virginica.map(pack_features_vector)
dataset_setosa = dataset_setosa.map(pack_features_vector)
# Dataset from Tensorflow
# train_dataset = train_dataset.map(pack_features_vector)
test_dataset = test_dataset.map(pack_features_vector)
# #train_dataset = train_dataset_iris_github.map(pack_features_vector)

train_dataset = dataset_versicolor

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
  for x, y in train_dataset:
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
    [6.3, 2.5, 5.0, 1.9]  # virginica
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset_sorted, training=False)

for i, logits in enumerate(predict_dataset_sorted):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

