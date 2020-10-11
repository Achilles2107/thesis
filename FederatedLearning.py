from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from pandas.plotting import scatter_matrix
from yellowbrick.target import FeatureCorrelation
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
print("imports ok")

daten = 'C:\\Users\\Stefan\\Nextcloud\\Thesisstuff\\Datensätze\\MachineLearningCSV\\MachineLearningCVE\\'


'''load datasets'''

file = daten + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'

df = pd.read_csv(file, header=None, low_memory=False, skiprows=1)

# Replace negative Infinity Values
df = df.replace([np.inf, -np.inf], 0).fillna(0)

# df.dropna(axis="rows", how="any", inplace=True)

print("Read {} rows.".format(len(df)))


df.columns = ["DestinationPort",
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

# display 5 rows
df.head()

ENCODING = 'utf-8'


def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v, round(100 * (s[v] / t), 2)))
    return "[{}]".format(",".join(result))


def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count > 100:
            print("** {}:{} ({}%)".format(col, unique_count, int(((unique_count) / total) * 100)))
        else:
            print("** {}:{}".format(col, expand_categories(df[col])))
            expand_categories(df[col])


# analyze the dataset see how it looks like
analyze(df)

df = df.reset_index()
dropped = df.dropna(inplace=True, axis=0)
print(dropped)
# check if NaN Values exist
result= df.isnull().sum().sum()
print('NAN Werte: ', result)

# NORMALIZATION of X Values
# print(df[0:5])
# separate array into input and output components
# X = df.values[:,0:66]
# Y = df.values[:,66]

X = df.values[:, 0:78]
Y = df.values[:, 78]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)

# TRANSFORM LABEL TO INTEGER
# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
Y_2 = le.fit_transform(Y)

print('LABEL ENCODING: /n')
print(Y_2)

Y = np.reshape(Y_2, (-1, 1))
enc = preprocessing.OneHotEncoder()
# 2. FIT
enc.fit(Y)
# 3. Transform
onehotlabels = enc.transform(Y).toarray()

X = normalizedX
Y = onehotlabels

print('ONE HOT ENCODING:')
print(Y)

# check if data looks normalized
df.head()

# remove rows with low variance, label has to be a number: 0,1,2..
# threshold_n = 0.99
# sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n) ))
# sel_var = sel.fit_transform(df)
# df = df[df.columns[sel.get_support(indices=True)]]
# df.head()

# Create a test/train split.  33% test
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42, shuffle=True)

# Create neural net
model = Sequential()
layer1 = Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')
layer2 = Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')
layer3 = Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')
layer4 = Dense(1, kernel_initializer='normal')
layer5 = Dense(Y.shape[1], activation='softmax')

model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, epochs=100)

# Measure accuracy
prediction = model.predict(x_test)
y_score = prediction
pred = np.argmax(prediction, axis=1)
y_eval = np.argmax(y_test, axis=1)
score = accuracy_score(y_eval, pred)
print("Validation score: {}".format(score))
print(history.history.keys())

# Confusion Matrix
print("MATRIX")
cfm = confusion_matrix(y_eval,pred)
print(cfm)

# Report
cmp = classification_report(y_eval,pred)
print(cmp)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()






