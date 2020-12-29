import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(frozen=True, slots=True, eq=False)
class IrisBatchOutput(tff.learning.BatchOutput):
    value = attr.ib()


class IrisModel(tff.learning.Model):

    def __init__(self):
        self._variables = IrisModel.create_iris_variables()

    @staticmethod
    def create_iris_variables():
        return collections.namedtuple(
            'IrisVariables', 'weights bias num_examples loss_sum accuracy_sum value')(
            weights=tf.Variable(
                lambda: tf.zeros(dtype=tf.float32, shape=(4, 10)),
                name='weights',
                trainable=True),
            bias=tf.Variable(
                lambda: tf.zeros(dtype=tf.float32, shape=(10)),
                name='bias',
                trainable=True),
            num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
            loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
            accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False),
            value=tf.Variable(lambda: tf.zeros(dtype=tf.int32, shape=(15)), name='value', trainable=False))

    @staticmethod
    def iris_forward_pass(variables, batch):
        y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
        predictions = tf.cast(tf.argmax(y, 1), tf.int32)
        print("THIS ARE OUR PREDICTIONS: " + str(predictions))

        flat_labels = tf.reshape(batch['y'], [-1])
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, flat_labels), tf.float32))

        num_examples = tf.cast(tf.size(batch['y']), tf.float32)

        variables.num_examples.assign_add(num_examples)
        variables.loss_sum.assign_add(loss * num_examples)
        variables.accuracy_sum.assign_add(accuracy * num_examples)
        variables.value.assign(predictions)

        return loss, predictions

    @staticmethod
    def get_local_iris_metrics(variables):
        return collections.OrderedDict(
            num_examples=variables.num_examples,
            loss=variables.loss_sum / variables.num_examples,
            accuracy=variables.accuracy_sum / variables.num_examples,
            value=variables.value)

    @staticmethod
    @tff.federated_computation
    def aggregate_iris_metrics_across_clients(metrics):
        return collections.OrderedDict(
            num_examples=tff.federated_sum(metrics.num_examples),
            loss=tff.federated_mean(metrics.loss, metrics.num_examples),
            accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples),
            value=tff.federated_collect(metrics.value))

    def get_variables(self):
        return self._variables

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum, self._variables.value
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 4], tf.float32),
            y=tf.TensorSpec([None], tf.int32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = IrisModel.iris_forward_pass(self._variables, batch)
        num_examples = tf.shape(batch['x'])[0]
        return IrisBatchOutput(
            loss=loss, predictions=predictions, num_examples=num_examples, value=self._variables.value)

    @tf.function
    def report_local_outputs(self):
        return IrisModel.get_local_iris_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return IrisModel.aggregate_iris_metrics_across_clients
