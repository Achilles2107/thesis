import tensorflow as tf
import tensorflow_federated as tff
import pathlib
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

split_data_path = test_path / 'Thesis/datasets/hinkelmann'

print("Split Data Patch: \n", split_data_path)

train_dataset_full = tf.data.experimental.make_csv_dataset(
                    str(split_data_path / 'out_normalized.csv'),
                    BATCH_SIZE,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1,
                    shuffle=False
)


# Packs features in a single array
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


features, labels = next(iter(train_dataset_full))
print(features, labels)

train_dataset_full = train_dataset_full.map(pack_features_vector)

print("packed dataset: \n", train_dataset_full)

# Create neural net

print('Creating Neural Network')

model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_shape=(78,), kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(50, kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(10, kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(1, kernel_initializer='normal'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset_full, verbose=2, epochs=10)


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


