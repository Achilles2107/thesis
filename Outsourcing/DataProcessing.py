import os
from sklearn.preprocessing import LabelEncoder
from Outsourcing import DataPreprocessing


class CreateDatasets:

    def __init__(self, url, filename, label_name, batch_size, title, shuffle_value=True,  column_names=None):
        self.url = url
        self.filename = filename
        self.label_name = label_name
        self.column_names = column_names
        self.batch_size = batch_size
        self.shuffle_value = shuffle_value
        self.title = title
        self.file_path = 0
        self.features = 0
        self.labels = 0
        self.dataset = 0

    # note ONLY csv files will work with
    # the dataset creation at the moment
    def create_file_list(self):
        file_list = os.listdir(self.url)
        print(file_list)
        return file_list

    # Create train dataset
    def create_iris_url_dataset(self):
        train_data = DataPreprocessing.PreprocessData(self.url, self.filename, self.label_name, self.batch_size,
                                                      self.title, self.shuffle_value, self.column_names)
        train_data.get_dataset_by_url()
        train_data.create_train_dataset()
        train_data.make_graph()
        train_data.map_dataset()
        train_dataset = train_data.dataset
        return train_dataset

    # Create test dataset
    def create_iris_local_dataset(self):
        test_data = DataPreprocessing.PreprocessData(self.url, self.filename, self.label_name, self.batch_size,
                                                     self.title, self.shuffle_value, self.column_names)
        test_data.get_local_dataset()
        test_data.create_train_dataset()
        test_data.make_graph()
        test_data.map_dataset()
        test_dataset = test_data.dataset
        return test_dataset


class CreateDatasetLists:

    def __init__(self):
        self.dataset_list = []

    def add_dataset_to_list(self, dataset):
        self.dataset_list.append(dataset)
        print("Added " + str(self.dataset_list[-1]) + " to list")

    def remove_dataset_from_list(self, index):
        self.dataset_list.remove(index)
        print("Removed " + self.dataset_list[index] + " from list")

    def show_list(self):
        for i in self.dataset_list:
            print(self.dataset_list[i] + "\n")

    def get_dataset_list(self):
        return self.dataset_list

    def get_dataset_at_index(self, index):
        return self.dataset_list[index]


# Feature handling and label endcoding
def iter_dataset(dataset):
    features, labels = next(iter(dataset))
    list = [features, labels]
    return list


def encode_label(labels):
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(labels)
        return label_ids
