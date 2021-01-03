import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_federated.python.simulation import client_data

np.random.seed(0)


class IrisClientData(client_data.ClientData):

    def __init__(self, filepath, skiprows=0, labelname='label'):
        self._filepath = filepath
        self._client_ids = list(range(10))
        self.skiprows = skiprows
        self.label_name = labelname

        self._pd_dataset = self._create_dataset()
        self._elements_per_client = len(self._pd_dataset) // len(self._client_ids)

    def _create_dataset(self):
        # client_id is unused at the moment
        df_test = pd.read_csv(
            self._filepath,
            names=["sepal_length", "sepal_width", "petal_length", "petal_width", str(self.label_name)],
            nrows=100, skiprows=self.skiprows)

        float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}

        int_cols = [c for c in df_test if df_test[c].dtype == "int64"]
        int32_cols = {c: np.int32 for c in int_cols}

        all_cols = {**int32_cols, **float32_cols}

        df = pd.read_csv(
            self._filepath,
            names=["sepal_length", "sepal_width", "petal_length", "petal_width", str(self.label_name)],
            engine='c', dtype=all_cols, skiprows=self.skiprows)

        # Set _element_type_structure

        # Shuffle the dataset because it is ordered in the file
        return df.sample(frac=1).reset_index(drop=True)

    def create_tf_dataset_for_client(self, client_id):
        # 150 entries complete iris
        if client_id not in self.client_ids:
            raise ValueError(
                "ID [{i}] is not a client in this ClientData. See "
                "property `client_ids` for the list of valid ids.".format(
                    i=client_id))

        # all entries divided pro client
        # Client 1: entries 0 to 14
        # Client 2: entries 15 to 29
        # ...
        elements = range(int(client_id) * self._elements_per_client,
                         int(client_id) * self._elements_per_client + self._elements_per_client)
        df = self._pd_dataset.take(elements)
        label = df.pop(str(self.label_name))
        return tf.data.Dataset.from_tensor_slices((df.values, label.values))

    @property
    def client_ids(self):
        return self._client_ids

    @property
    def element_type_structure(self):
        return self._element_type_structure

    @property
    def dataset_computation(self):
        raise NotImplementedError("b/162106885")
