from outsourcing import DataPreprocessing
import pathlib

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

# File for datasets for use in multiple models


class IrisDatasets:

    # Iris datasets
    # Filepaths

    saved_model_path = root_project_path / '/storage/iris_model/'
    dataset_path_local = root_project_path / '/datasets/iris_classification/'
    logfile_path = root_project_path / 'datasets/iris_classification/logs/'
    #split_train_data_path = root_project_path / '/datasets/iris_classification/split/train/'
    split_test_data_path = root_project_path / '/datasets/iris_classification/split/test/'

    split_train_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\Thesis\\datasets\\iris_classification\\split\\train\\'

    # Urls and paths
    github_dataset = dataset_path_local / 'iris_training02.csv'
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
    test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

    # Parameter
    batch_size = 120

    # Create Traindata
    train_data = DataPreprocessing.PreprocessData(
        train_dataset_url, 'iris_training.csv', label_name, batch_size,
        'Iris Train CSV Tensorflow', True, column_names)
    train_data.get_dataset_by_url()
    train_data.create_train_dataset()
    train_data.make_graph()
    train_data.map_dataset()
    train_dataset = train_data.dataset

    # Create Test Dataset
    test_data = DataPreprocessing.PreprocessData(test_url, 'iris_test.csv', label_name, batch_size,
                                                 'Iris Test CSV Tensorflow', False, column_names)

    test_data.get_dataset_by_url()
    test_data.create_train_dataset()
    test_data.make_graph()
    test_data.map_dataset()
    test_dataset = test_data.dataset

    # Create Split Datasets

    split01_data = DataPreprocessing.PreprocessData(split_train_data_path, '1.csv', label_name, batch_size,
                                                 'Split01 Train CSV Tensorflow', False, column_names)

    split01_data.get_local_dataset()
    split01_data.create_train_dataset()
    split01_data.make_graph()
    split01_data.map_dataset()
    split01_dataset = split01_data.dataset

    split02_data = DataPreprocessing.PreprocessData(split_train_data_path, '2.csv', label_name, batch_size,
                                                    'Split02 Train CSV Tensorflow', False, column_names)

    split02_data.get_local_dataset()
    split02_data.create_train_dataset()
    split02_data.make_graph()
    split02_data.map_dataset()
    split02_dataset = split02_data.dataset

    split03_data = DataPreprocessing.PreprocessData(split_train_data_path, '3.csv', label_name, batch_size,
                                                    'Split03 Train CSV Tensorflow', False, column_names)

    split03_data.get_local_dataset()
    split03_data.create_train_dataset()
    split03_data.make_graph()
    split03_data.map_dataset()
    split03_dataset = split01_data.dataset

