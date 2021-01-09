from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import pathlib
import csv
import tensorflow as tf

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

print("imports ok")


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

# Run TensorFlow on CPU only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


output_path = pathlib.Path('C:/Users/Stefan/PycharmProjects/Thesis/Hinkelmann')
daten = pathlib.Path('C:/Users/Stefan/Nextcloud/Thesisstuff/Datensätze/MachineLearningCSV/MachineLearningCVE/')


'''load Datasets'''

file = daten / 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'

# Can be used to get column names
with open(file, 'r') as f:
    d_reader = csv.DictReader(f)

    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames
    print(str(headers))

df = pd.read_csv(file, header=None, skiprows=1, nrows=500)  # type: pd.DataFrame

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

column_names = df.columns
label_name = column_names[-1]

# display 5 rows
print(df.head())

ENCODING = 'utf-8'

# analyze the dataset see how it looks like
analyze(df)

df = df.reset_index()
dropped = df.dropna(inplace=True, axis=0)
print(dropped)
# check if NaN Values exist
result = df.isnull().sum().sum()
print('NAN Werte: ', result)

# NORMALIZATION of X Values
print(df[0:5])
# separate array into input and output components
# X = df.values[:,0:66]
# Y = df.values[:,66]

X = df.values[:, 0:78]
Y = df.values[:, 78]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)

print("normalized X values: \n", X)

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


# Seems to have no effect - fix when time left or test with more rows
# remove rows with low variance, label has to be a number: 0,1,2..
# threshold_n = 0.99
# sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n) ))
# sel_var = sel.fit_transform(df)
# df = df[df.columns[sel.get_support(indices=True)]]
# print(df.head())

# not working yet
# df.to_csv(output_path / 'out.csv', index=False)
