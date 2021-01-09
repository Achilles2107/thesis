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
    split_train_data_path = root_project_path / '/datasets/iris_classification/split/train/'
    split_test_data_path = root_project_path / '/datasets/iris_classification/split/test/'

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
    train_data = DataPreprocessing.PreprocessData(train_dataset_url, 'iris_training.csv', label_name, batch_size,
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

