from outsourcing.Datasets import IrisDatasets
from outsourcing.DataProcessing import *
from outsourcing.CustomMetrics import *
import tensorflow as tf

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


# Define a keras model
# input_shape will generate an additional input layer
# tf.keras.layers.InputLayer() can be used as an alternative way
# to create an input layer
keras_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# Define metrics used in the training
# Metrics used here are from CustomMetrics.py
# alternative use tf.keras.metrics.<metric_name>()
METRICS = [
    recall,
    specificity,
    precision,
    ["accuracy"],
    print_y_true,
    print_y_pred
]

# Compile the defined model above and declare optimizer, loss
# and metrics
keras_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=METRICS
                    )


# Display the model's architecture
keras_model.summary()

# Start training and define optional validation dataset
history = keras_model.fit(
                                train_features, train_labels,
                                validation_data=(test_features, test_labels),
                                epochs=epochs
                              )

# Get mean training accuracy and plot metrics
mean_training_accuracy(history, "accuracy")
subplot_metrics(history, "accuracy", "precision")
subplot_metrics(history, "recall", "specificity")

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

predictions = keras_model.predict(train_features)
print(predictions)

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = keras_model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

train_acc = keras_model.evaluate(train_features, train_labels)
test_acc = keras_model.evaluate(test_features, test_labels)

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
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
predictions = keras_model(predict_dataset_sorted, training=False)

for i, logits in enumerate(predict_dataset_sorted):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
