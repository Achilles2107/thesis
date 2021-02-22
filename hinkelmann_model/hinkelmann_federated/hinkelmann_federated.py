import tensorflow as tf
import tensorflow_federated as tff
import pathlib
import collections
import pandas as pd
import numpy as np

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

print("imports ok")

# Packs features in a single array
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


# Run TensorFlow on CPU only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

output_path = pathlib.Path('hinkelmann_model')

column_names = ["DestinationPort",
              "FlowDuration",
              "TotalFwdPackets",
              "TotalBackwardPackets",
              "TotalLengthofFwdPackets",
              "TotalLengthofBwdPackets",
              "FwdPacketLengthMax",
              "FwdPacketLengthMin",
              "FwdPacketLengthMean",
              "FwdPacketLengthStd",
              "BwdPacketLengthMax",
              "BwdPacketLengthMin",
              "BwdPacketLengthMean",
              "BwdPacketLengthStd",
              "FlowBytes/s",
              "FlowPackets/s",
              "FlowIATMean",
              "FlowIATStd",
              "FlowIATMax",
              "FlowIATMin",
              "FwdIATTotal",
              "FwdIATMean",
              "FwdIATStd",
              "FwdIATMax",
              "FwdIATMin",
              "BwdIATTotal",
              "BwdIATMean",
              "BwdIATStd",
              "BwdIATMax",
              "BwdIATMin",
              "FwdPSHFlags",
              "BwdPSHFlags",
              "FwdURGFlags",
              "BwdURGFlags",
              "FwdHeaderLength",
              "BwdHeaderLength",
              "FwdPackets/s",
              "BwdPackets/s",
              "MinPacketLength",
              "MaxPacketLength",
              "PacketLengthMean",
              "PacketLengthStd",
              "PacketLengthVariance",
              "FINFlagCount",
              "SYNFlagCount",
              "RSTFlagCount",
              "PSHFlagCount",
              "ACKFlagCount",
              "URGFlagCount",
              "CWEFlagCount",
              "ECEFlagCount",
              "Down/UpRatio",
              "AveragePacketSize",
              "AvgFwdSegmentSize",
              "AvgBwdSegmentSize",
              "FwdHeaderLength2",
              "FwdAvgBytes/Bulk",
              "FwdAvgPackets/Bulk",
              "FwdAvgBulkRate",
              "BwdAvgBytes/Bulk",
              "BwdAvgPackets/Bulk",
              "BwdAvgBulkRate",
              "SubflowFwdPackets",
              "SubflowFwdBytes",
              "SubflowBwdPackets",
              "SubflowBwdBytes",
              "Init_Win_bytes_forward",
              "Init_Win_bytes_backward",
              "act_data_pkt_fwd",
              "min_seg_size_forward",
              "ActiveMean",
              "ActiveStd",
              "ActiveMax",
              "ActiveMin",
              "IdleMean",
              "IdleStd",
              "IdleMax",
              "IdleMin",
              "Label"]

label_name = column_names[-1]
feature_names = column_names[:-1]

# HERE STARTS THE FEDERATED LEARNING
NUM_EPOCHS = 5
BATCH_SIZE = 5000
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

levels_up = 1
test_path = root_project_path.parents[levels_up-1]

print("ROOT PROJECT PATH: \n", test_path)

split_data_path = test_path / 'datasets/hinkelmann'

print("Split Data Patch: \n", split_data_path)

train_dataset_full = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'out.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)


train_dataset01 = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'split1.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)

train_dataset02 = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'split2.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)

train_dataset03 = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'split3.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)

test_dataset01 = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'split4.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)

print(train_dataset01)

features, labels = next(iter(train_dataset_full))

train_dataset_full = train_dataset_full.map(pack_features_vector)
# train_dataset01 = train_dataset01.map(pack_features_vector)
# train_dataset02 = train_dataset02.map(pack_features_vector)
# train_dataset03 = train_dataset03.map(pack_features_vector)
# test_dataset01 = test_dataset01.map(pack_features_vector)


# print("train dataset 01", train_dataset01)
#
# federated_train_data = [train_dataset01, train_dataset02, train_dataset03]

federated_train_data = [train_dataset_full, train_dataset_full, train_dataset_full]

print("federated data: ", federated_train_data)

# Create neural net

# Metrics
METRICS = [
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.SensitivityAtSpecificity(0.5),
        tf.keras.metrics.SpecificityAtSensitivity(0.5),
        tf.keras.metrics.Accuracy()
]

print('Creating Neural Network')

# Measure accuracy

# CustomMetrics.mean_training_accuracy(history, "accuracy")
# CustomMetrics.plot_metric(history, "recall")
# CustomMetrics.plot_metric(history, "precision")
# CustomMetrics.plot_metric(history, "sensitivity_at_specificity")
# CustomMetrics.plot_metric(history, "specificity_at_sensitivity")
# CustomMetrics.subplot_metrics(history, "accuracy", "loss")
#
#
# prediction = model.predict(x_test)
# y_score = prediction
# pred = np.argmax(prediction, axis=1)
# y_eval = np.argmax(y_test, axis=1)
# score = accuracy_score(y_eval, pred)
# print("Validation score: {}".format(score))
# print(history.history.keys())
#
# # Confusion Matrix
# print("MATRIX")
# cfm = confusion_matrix(y_eval, pred)
# print(cfm)
#
# # Report
# cmp = classification_report(y_eval, pred)
# print(cmp)


def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_shape=(78,), kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(50, kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(10, kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(1, kernel_initializer='normal'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      # Input information on what shape the input data will have
      # Must be from type tf.Type or tf.TensorSpec
      input_spec=train_dataset01.element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.Accuracy()])


keras_model = create_keras_model()


fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0))  # for Global model


state = fed_avg.initialize()

epochs = 10

# Start Federated Learning process
for round_num in range(1, epochs):
    state, metrics = fed_avg.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

