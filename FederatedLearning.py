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
import os
print("imports ok")

# Run TensorFlow on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

daten = 'C:\\Users\\Stefan\\Nextcloud\\Thesisstuff\\Datensätze\\MachineLearningCSV\\MachineLearningCVE\\'

# daten = '/home/stefan/daten/'

'''load datasets'''

file = daten + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'

df = pd.read_csv(file, header=None, low_memory=False, skiprows=1, nrows=50)

# Replace negative Infinity Values
df = df.replace([np.inf, -np.inf], 0).fillna(0)

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

print('Creating Neural Network')

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
cfm = confusion_matrix(y_eval, pred)
print(cfm)

# Report
cmp = classification_report(y_eval, pred)
print(cmp)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Ausgangsmodell
a1 = layer1.get_weights()
a2 = layer2.get_weights()
a3 = layer3.get_weights()
a4 = layer4.get_weights()
a5 = layer5.get_weights()

print(a4)

layer1.set_weights(a1)
layer2.set_weights(a2)
layer3.set_weights(a3)
layer4.set_weights(a4)
layer5.set_weights(a5)
print(layer4.get_weights())

# Brute Force
w1 = layer1.get_weights()
w2 = layer2.get_weights()
w3 = layer3.get_weights()
w4 = layer4.get_weights()
w5 = layer5.get_weights()

print(w4)

# XSS
w1_1 = layer1.get_weights()
w2_1 = layer2.get_weights()
w3_1 = layer3.get_weights()
w4_1 = layer4.get_weights()
w5_1 = layer5.get_weights()

print(w4_1)

#SQL
w1_2 = layer1.get_weights()
w2_2 = layer2.get_weights()
w3_2 = layer3.get_weights()
w4_2 = layer4.get_weights()
w5_2 = layer5.get_weights()

print(w4_2)

# calculate weigths of 3 models
weight_list = [w1[0], w2[0], w3[0], w4[0], w5[0]]
weight_list2 = [w1_1[0], w2_1[0], w3_1[0], w4_1[0], w5_1[0]]
weight_list3 = [w1_2[0], w2_2[0], w3_2[0], w4_2[0], w5_2[0]]

weight_1 = []
weight_2 = []
weight_3 = []
weight_4 = []
weight_5 = []

for i in range(len(weight_list[0])):
    row = []
    for j in range(len(weight_list[0][0])):
        row.append((weight_list[0][i][j] + weight_list2[0][i][j] + weight_list3[0][i][j]) / 3)
    weight_1.append(row)

for i in range(len(weight_list[1])):
    row = []
    for j in range(len(weight_list[1][0])):
        row.append((weight_list[1][i][j] + weight_list2[1][i][j] + weight_list3[1][i][j]) / 3)
    weight_2.append(row)

for i in range(len(weight_list[2])):
    row = []
    for j in range(len(weight_list[2][0])):
        row.append((weight_list[2][i][j] + weight_list2[2][i][j] + weight_list3[2][i][j]) / 3)
    weight_3.append(row)

for i in range(len(weight_list[3])):
    row = []
    for j in range(len(weight_list[3][0])):
        row.append((weight_list[3][i][j] + weight_list2[3][i][j] + weight_list3[3][i][j]) / 3)
    weight_4.append(row)

for i in range(len(weight_list[4])):
    row = []
    for j in range(len(weight_list[4][0])):
        row.append((weight_list[4][i][j] + weight_list2[4][i][j] + weight_list3[4][i][j]) / 3)
    weight_5.append(row)

weight_1 = np.array(weight_1)
weight_1 = weight_1.astype('float32')
weight_2 = np.array(weight_2)
weight_2 = weight_2.astype('float32')
weight_3 = np.array(weight_3)
weight_3 = weight_3.astype('float32')
weight_4 = np.array(weight_4)
weight_4 = weight_4.astype('float32')
weight_5 = np.array(weight_5)
weight_5 = weight_5.astype('float32')


#calculate bias of 3 models
bias_list = [w1[1],w2[1],w3[1],w4[1],w5[1]]
bias_list2 = [w1_1[1],w2_1[1],w3_1[1],w4_1[1],w5_1[1]]
bias_list3 = [w1_2[1],w2_2[1],w3_2[1],w4_2[1],w5_2[1]]

bias_1 = []
bias_2 = []
bias_3 = []
bias_4 = []
bias_5 = []


bias_1.append( ( bias_list[0] + bias_list2[0] + bias_list3[0]) / 3 )
bias_1 = np.array(bias_1[0])
bias_1 = bias_1.astype('float32')

bias_2.append( ( bias_list[1] + bias_list2[1] + bias_list3[1]) / 3 )
bias_2 = np.array(bias_2[0])
bias_2 = bias_2.astype('float32')

bias_3.append( ( bias_list[2] + bias_list2[2] + bias_list3[2]) / 3 )
bias_3 = np.array(bias_3[0])
bias_3 = bias_3.astype('float32')

bias_4.append( ( bias_list[3] + bias_list2[3] + bias_list3[3]) / 3 )
bias_4 = np.array(bias_4[0])
bias_4 = bias_4.astype('float32')

bias_5.append( ( bias_list[4] + bias_list2[4] + bias_list3[4]) / 3 )
bias_5 = np.array(bias_5[0])
bias_5 = bias_5.astype('float32')

# bias_1 = np.reshape(bias_1, (10,))
# bias_2 = np.reshape(bias_2, (50,))
# bias_3 = np.reshape(bias_3, (10,))
# bias_4 = np.reshape(bias_4, (1, ))
# bias_5 = np.reshape(bias_5, (2, ))

# set new weights
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]

x1= weight_1 #weights
y1= bias_1 #array of biases
x2= weight_2 #weights
y2= bias_2 #array of biases
x3= weight_3 #weights
y3= bias_3 #array of biases
x4= weight_4 #weights
y4= bias_4 #array of biases
x5= weight_5 #weights
y5= bias_5 #array of biases

l1.append(x1)
l1.append(y1)
l2.append(x2)
l2.append(y2)
l3.append(x3)
l3.append(y3)
l4.append(x4)
l4.append(y4)
l5.append(x5)
l5.append(y5)

layer1.set_weights(l1)
layer2.set_weights(l2)
layer3.set_weights(l3)
layer4.set_weights(l4)
layer5.set_weights(l5)

# Build federated model
model = Sequential()
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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



