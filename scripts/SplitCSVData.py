import pandas as pd
import numpy as np
from sklearn.utils import shuffle


dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = dataset_path_local + 'split\\'


def read_csv(path, filename):
    df = pd.read_csv(path + filename)
    return df


def shuffle_csv(df, rounds):
    df_shuffle = df
    for i in range(rounds):
        df_shuffle = shuffle(df_shuffle)
        print('shuffle nr. ' + str(i + 1))
        print(df_shuffle)
    return df_shuffle


def split_csv(df, parts):
    df = np.array_split(df, parts)
    return df


def write_to_csv(df, parts, path):
    for i in range(parts):
        df[i].to_csv(path + str(i + 1) + '.csv', index=False)


# df = read_csv(dataset_path_local, "iris_test.csv")
# df_shuffled = shuffle_csv(df, 20)
# df_splitted = split_csv(df_shuffled, 4)
# write_to_csv(df_splitted, 4, split_data_path + 'test\\')
# print("done")

with open(dataset_path_local + 'split\\test\\3.csv', 'r') as f1:
    original = f1.read()

with open(dataset_path_local + 'split\\test\\123.csv', 'a') as f2:
    f2.write('\n')
    f2.write(original)