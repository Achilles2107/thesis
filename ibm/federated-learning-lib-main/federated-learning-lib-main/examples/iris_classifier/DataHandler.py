import numpy as np

# imports from ibmfl
from ibmfl.data.data_handler import DataHandler
from ibmfl.exceptions import FLException


class MnistKerasDataHandler(DataHandler):
    """
    Data handler for MNIST dataset.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        # `data_config` loads anything inside the `info` part of the `data` section.
        if data_config is not None:
            # this example assumes the local dataset is in .npz format, so it searches for it.
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        self.channels_first = channels_first

        if self.file_name is None:
            raise FLException('No data file name is provided to load the dataset.')
        else:
            try:
                data_train = np.load(self.file_name)
                self.x_train = data_train['x_train']
                self.y_train = data_train['y_train']
                self.x_test = data_train['x_test']
                self.y_test = data_train['y_test']
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.file_name)
            self.preprocess_data()

    def get_data(self):
        """
        Gets pre-processed mnist training and testing data.

        :return: training and testing data
        :rtype: `tuple`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def preprocess_data(self):
        """
        Preprocesses the training and testing dataset.

        :return: None
        """
        num_classes = 10
        img_rows, img_cols = 28, 28
        if self.channels_first:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = np.eye(num_classes)[self.y_train]
        self.y_test = np.eye(num_classes)[self.y_test]


