# Linear regression y=mx+b
import tensorflow as tf
import matplotlib.pyplot as plt

# Fetching Dataset
from tensorflow.keras.datasets import boston_housing

class SimpleLinearRegression:
    def __init__(self, initializer='random'):
        if initializer == 'ones':
            self.var = 1.
        elif initializer == 'zeros':
            self.var = 0.
        elif initializer == 'random':
            self.var = tf.random.uniform(shape=[], minval=0., maxval=1.)

        self.m = tf.Variable(1., shape=tf.TensorShape(None))
        self.b = tf.Variable(self.var)

    def mse(self, true, predicted):
        return tf.reduce_mean(tf.square(true - predicted))


    def predict(self, x):
        return tf.reduce_sum(self.m * x, 1) + self.b

    def update(self, X, y, learning_rate):
        with tf.GradientTape(persistent=True) as g:
            loss = self.mse(y, self.predict(X))

        print("Loss: ", loss)

        dy_dm = g.gradient(loss, self.m)
        dy_db = g.gradient(loss, self.b)

        self.m.assign_sub(learning_rate * dy_dm)
        self.b.assign_sub(learning_rate * dy_db)

    def train(self, X, y, learning_rate=0.01, epochs=5):

        if len(X.shape) == 1:
            X = tf.reshape(X, [X.shape[0], 1])

        self.m.assign([self.var] * X.shape[-1])

        for i in range(epochs):
            print("Epoch: ", i)

            self.update(X, y, learning_rate)


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

mean_label = y_train.mean(axis=0)
std_label = y_train.std(axis=0)
mean_feat = x_train.mean(axis=0)
std_feat = x_train.std(axis=0)
x_train = (x_train-mean_feat)/std_feat
y_train = (y_train-mean_label)/std_label

linear_model = SimpleLinearRegression('zeros')
linear_model.train(x_train, y_train, learning_rate=0.1, epochs=50)

# standardize
x_test = (x_test-mean_feat)/std_feat
# reverse standardization
pred = linear_model.predict(x_test)
pred *= std_label
pred += mean_label

# plotting points as a scatter plot
plt.scatter(x_train, y_train, label="stars", color="green",
            marker="*", s=30)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title("Linear Regression")

plt.show()
