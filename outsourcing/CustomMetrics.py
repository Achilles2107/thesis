import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File for custom metrics to use with keras


# Source: https://en.wikipedia.org/wiki/Precision_and_recall
# Precision is the number of correct results divided by the number of
# all returned results
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


#  Recall is the number of correct results divided
#  by the number of results that should have been returned
#  in binary classification recall is called sensitivity
#  It can be viewed as the probability that a
#  relevant document is retrieved by the query
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Source: https://medium.com/@mostafa.m.ayoub/customize-your-keras-metrics-44ac2e2980bd
# Source: https://dzone.com/articles/ml-metrics-sensitivity-vs-specificity-difference
# Specificity is defined as the proportion of actual negatives,
# which got predicted as the negative (or true negative).
# This implies that there will be another proportion of actual negative,
# which got predicted as positive and could be termed as false positives.
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Average training accuracy
# Must be called after training
def mean_training_accuracy(model, metric):
    model.history[str(metric)]
    mean_acc = np.mean(model.history[str(metric)])
    mean_acc = round(float(mean_acc), 2)
    return print("Average training accuracy: " + str(100*mean_acc) + "%")


def print_y_true(y_true, y_pred):
       return y_true


def print_y_pred(y_true, y_pred):
        return y_pred


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def plot_metric(model, metric_name):
    plt.plot(model.history[str(metric_name)])
    plt.title(str(metric_name))
    plt.show()


def subplot_metrics(model, metric_name1, metric_name2):
    fig, axs = plt.subplots(2)
    fig.suptitle(str(metric_name1) + " & " + str(metric_name2))
    axs[0].plot(model.history[str(metric_name1)])
    axs[1].plot(model.history[str(metric_name2)])
    plt.show()


def plot_metric_val(model, metric_name1, metric_name2):
    plt.title(str(metric_name1) + " & " + str(metric_name2))
    plt.plot(model.history[str(metric_name1)], label=metric_name1)
    plt.plot(model.history[str(metric_name2)], label=metric_name2)
    plt.legend()
    plt.show()


