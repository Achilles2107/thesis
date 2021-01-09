from outsourcing import CustomMetrics
from outsourcing.Datasets import IrisDatasets
from outsourcing.DataProcessing import *
from outsourcing.CustomMetrics import *
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

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])


METRICS = [
    recall,
    specificity,
    precision,
    ["accuracy"]
]

# Compile the defined model above and declare optimizer, loss
# and metrics
model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=METRICS
                    )

print(model.metrics)

# Display the model's architecture
model.summary()

# load model
# model.load_weights(checkpoint_path)

training_history = model.fit(train_features, train_labels, epochs=epochs, validation_data=(test_features, test_labels),
                             callbacks=cp_callback)


CustomMetrics.mean_training_accuracy(training_history, "accuracy")

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# print("Plotting metrics")
# CustomMetrics.subplot_metrics(training_history, "recall", "specificity")
# CustomMetrics.plot_metric(training_history, "precision")
# CustomMetrics.subplot_metrics(training_history, "accuracy", "loss")
# print("done")

model.save(path / "/keras_model/", overwrite=True, include_optimizer=True)

train_acc = model.evaluate(train_features, train_labels)
test_acc = model.evaluate(test_features, test_labels)


