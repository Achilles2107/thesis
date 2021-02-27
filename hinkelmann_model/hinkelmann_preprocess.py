from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import pathlib
import csv
import sys

# Uncommend to print everything
# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

print("imports ok")

new_file = False

# force numpy and pandas to print every column
# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
# pd.set_option('display.max_columns', None)

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


def normalize(dataset):
    data_norm = ((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    #data_norm["label"] = dataset["label"]
    return data_norm


output_path = pathlib.Path('hinkelmann_model')
# path to datasets in .csv files
daten = pathlib.Path('C:/Users/Stefan/Nextcloud/Thesisstuff/Datensätze/MachineLearningCSV/MachineLearningCVE/')

# name of the used dataset
file = daten / 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'

# Can be used to get column names
with open(file, 'r') as f:
    d_reader = csv.DictReader(f)

    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames
    print(str(headers))

# df_test = pd.read_csv(file, low_memory=False, header=None)
#
# float_cols = [c for c in df_test if df_test[c].dtype == "double"]
# float_cols = {c: np.float32 for c in float_cols}
#
# int_cols = [c for c in df_test if df_test[c].dtype == "int64"]
# int32_cols = {c: np.int32 for c in int_cols}
#
# all_cols = {**int32_cols, **float32_cols}
#
# df = pd.read_csv(file, header=None, engine='c', dtype=all_cols, low_memory=False)
#
# df = pd.read_csv(file, header=None, skiprows=1, nrows=5000) # nrows limits the number of rows

if new_file is True:
    df = pd.read_csv(file, header=None, low_memory=False)

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

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # display 5 rows
    print(df.head())

    ENCODING = 'utf-8'

    # analyze the dataset see how it looks like
    analyze(df)

    # NORMALIZATION of X Values
    print(df[0:5])
    # separate array into input and output components
    # X = df.values[:,0:66]
    # Y = df.values[:,66]

    X = df.values[:, 0:78]
    Y = df.values[:, 78]

    print("DF: ", df)

    label = df.values[:, 78]

    # TRANSFORM LABEL TO INTEGER
    # 1. INSTANTIATE
    # encode labels with value between 0 and n_classes-1.
    le = preprocessing.LabelEncoder()
    # 2/3. FIT AND TRANSFORM
    # use df.apply() to apply le.fit_transform to all columns
    label_onehot = le.fit_transform(Y)

    print("label: ", label_onehot)
    print("label test", df.head())
    df.pop("Label")

    # print("removing first row, uncomment if no column names in first row")
    # df.drop(df.index[0], inplace=True)
    # print("done", df.head())

    # df = normalize(df)
    print("DF normalized: ", df)
    df.insert(78, "Label", label_onehot)
    print("labels are back: \n", df.head())

    # check if NaN Values exist
    result = df.isnull().sum().sum()
    print('NAN Werte: ', result)
    # df = df.reset_index()
    dropped = df.dropna(inplace=True, axis=0)
    print("dropped: \n", dropped)
    df.dropna(inplace=True, axis=0)

    # Seems to have no effect - fix when time is left or test with more rows
    # remove rows with low variance, label has to be a number: 0,1,2..
    # threshold_n = 0.99
    # sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
    # sel_var = sel.fit_transform(df)
    # print(df.head())

    print("Replacing , with .")
    df.replace(',', '.', inplace=True, regex=True)
    print("done", df.head())

    print("Dtypes: \n", df.dtypes)

    # print df into out.csv for further use
    df.to_csv(root_project_path / output_path / 'out.csv', index=False, header=False)

    out_path = root_project_path / output_path / 'out.csv'

    print("Written to: \n", out_path)

else:
    out_path = root_project_path / output_path / 'out.csv'

    df = pd.read_csv(out_path, header=None, skiprows=1)

    index = df.index
    number_of_rows = len(index)

    print(str(number_of_rows) + " rows read")

    print("DF START HEAD \n", df.head())

    df[df < 0] = 0
    df[df == np.inf] = 0
    df.dropna()

    print("INPUT DF: \n", df.head())

    print(str(number_of_rows) + " rows now")

    # scaler = preprocessing.normalize(df, norm='l1', axis=0, copy=False, return_norm=False)
    # # scaler.transform(df)

    # scaler = preprocessing.Normalizer(df)
    # scaler.fit(df)
    # scaler.transform(df)
    df_label = df[78].values
    df.pop(df.index[78])

    print("DF: \n", df.head())

    print("LABEL DF: \n", df_label)

    # Min-Max normalization
    normalized_df = (df - df.min()) / (df.max() - df.min())

    # scaler = preprocessing.MinMaxScaler()
    # df = scaler.fit_transform(df.values.reshape(-1, 1))

    #print("DF_LABEL DTYPES \n", df_label.dtypes)

    normalized_df[78] = df_label

    print("test : \n", normalized_df.head())

    print(str(number_of_rows) + " rows now")
    print("Normalized: \n", normalized_df.head())

    print("DTYPES: \n", normalized_df.dtypes)

    # print df into out.csv for further use
    normalized_df.to_csv(root_project_path / output_path / 'out_normalized.csv', index=False, header=False)

    print("Written to: \n", out_path)
