import nest_asyncio
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

nest_asyncio.apply()


@tff.federated_computation
def hello_world():
  return 'Hello, World!'


print(hello_world())

federated_float_on_clients = tff.type_at_clients(tf.float32)

print(str(federated_float_on_clients.member))

print(str(federated_float_on_clients.placement))

print(str(federated_float_on_clients))

print(federated_float_on_clients.all_equal)

batch = [1, 2, 3, 4, 5, 6, 7, 8, 9]

flat_labels = tf.reshape(batch, [-1])

print(flat_labels)


