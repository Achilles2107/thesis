import os
from outsourcing.datasets import IrisDatasets
from outsourcing.iris_classification.data_processing import *
from outsourcing.custom_metrics import *
import pathlib as path
from pathlib import Path

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
cwd = path.Path.cwd().parent
path = Path(cwd / 'storage/iris_model')

# Path for saving weights
checkpoint_path = path / "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Parameter
# Training iterations
epochs = 200
# Number of classes
num_classes = 3

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# datasets
# Get datasets from IrisDataset class
train_dataset = IrisDatasets.train_dataset
print(train_dataset)

test_dataset = IrisDatasets.test_dataset
print(test_dataset)

# Print datasets
print("Train Dataset: \n")
print(get_features_labels(train_dataset))
print("Test Dataset: \n")
print(get_features_labels(train_dataset))

# Decode and put labels in a binary matrix and get features
# Binary matrix is need as we are classifying to more than two classes
train_features, train_labels = decode_label(train_dataset, num_classes)
test_features, test_labels = decode_label(test_dataset, num_classes)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# Make Predictions
predictions = model(train_features, train_labels)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(train_labels))

# Define the loss and gradient function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, train_features, train_labels, training=False)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


loss_value, grads = grad(model, train_features, train_labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, train_features, train_labels, training=True).numpy()))

# Training Loop
# Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []


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

print("Train Dataset")
for (x, y) in train_dataset:
    print("X: " + str(x), "Y: " + str(y))

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()


for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

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
    [6.3, 2.5, 5.0, 1.9]     # virginica
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset_sorted, training=False)

for i, logits in enumerate(predict_dataset):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
