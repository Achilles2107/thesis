import os
import tensorflow as tf
from Outsourcing import CustomMetrics
from Outsourcing import Datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
saved_model_path = '/Storage\\IrisModel\\'

# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Parameter
batch_size = 120
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Datasets
iris_datasetlist = Datasets.iris_datalist

train_dataset = iris_datasetlist.get_dataset_at_index(0)
train_dataset02 = iris_datasetlist.get_dataset_at_index(2)
test_dataset = iris_datasetlist.get_dataset_at_index(1)

train_features, train_labels = next(iter(train_dataset))
test_features, test_labels = next(iter(train_dataset))

y_train = tf.keras.utils.to_categorical(train_features, num_classes=None)
y_test = tf.keras.utils.to_categorical(test_features, num_classes=None)


print("Y_TRAIN")
print(y_train.shape)

y_train.reshape(30, 3)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Accuracy(name="acc")
]


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3)
])


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=METRICS)

print(model.metrics)

# Display the model's architecture
model.summary()

# load model
model.load_weights(checkpoint_path)

training_history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=cp_callback)


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

model.save(saved_model_path + "\\keras_model\\", overwrite=True, include_optimizer=True)

train_acc = model.evaluate(train_dataset)
test_acc = model.evaluate(test_dataset)


