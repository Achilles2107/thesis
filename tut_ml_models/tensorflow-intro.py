import tensorflow as tf

print(tf.version)

string = tf.Variable("This is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

rank1_tensor = tf.Variable(["Test", "OK", "Hans"], tf.string) # Dimension 1 shape(1, )
rank2_tensor = tf.Variable([["Test", "OK"], ["test", "yes"]], ["test", "yes"], tf.string) # Dimension 2 shape(3, 3)

print("rank1_tensor.shape: ")
print(rank1_tensor.shape)

print("rank2_tensor.shape: ")
print(rank2_tensor.shape)

tensor1 = tf.ones([1, 2, 3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,3]

# The numer of elements in the reshaped tensor MUST match the number in the original

print(tensor1)
print(tensor2)
print(tensor3)
# Notice the changes in shape


