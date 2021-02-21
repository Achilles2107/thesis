import pandas as pd
import numpy as np
import pathlib
from sklearn.utils import shuffle

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

dataset_path_local = pathlib.Path('/datasets/hinkelmann/')
split_data_path = root_project_path / dataset_path_local / "split"
path = pathlib.Path('C:/Users/Stefan/PycharmProjects/Thesis/hinkelmann/')

root_path = pathlib.Path('C:/Users/Stefan/PycharmProjects/Thesis/datasets/hinkelmann/split')


def read_csv(path, filename):
    df = pd.read_csv(path + filename, skiprows=1)
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


df = read_csv(str(root_path), "\\out.csv")
df_shuffled = shuffle_csv(df, 20)
df_splitted = split_csv(df_shuffled, 4)
write_to_csv(df_splitted, 4, str(root_path))
print("done")

# with open(dataset_path_local + 'split\\test\\3.csv', 'r') as f1:
#     original = f1.read()
#
# with open(dataset_path_local + 'split\\test\\123.csv', 'a') as f2:
#     f2.write('\n')
#     f2.write(original)
