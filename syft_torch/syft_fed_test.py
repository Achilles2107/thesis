import numpy
import sklearn
from sklearn import datasets
import syft
import diffprivlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

data, labels = datasets.load_iris(return_X_y=True)
data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels)

def sweep(eps=[0.00001, 0.001, 0.01, 10, 1000, 10000]):
  accuracy = []
  for i in range(len(eps)):
    model = diffprivlib.models.GaussianNB(epsilon=eps[i])
    model.fit(data_train, labels_train)
    accuracy.append(sklearn.metrics.accuracy_score(labels_test, model.predict(data_test)))
  return accuracy

eps=[0.00001, 0.001, 0.01, 10, 1000, 10000] # Try changing these values to see how the accuracy plot changes!

accuracy = sweep(eps)


plt.figure() # notice that the accuracy for each value changes every time you run it with the same values
plt.semilogx(eps, accuracy) # this is because random noise is... well, random, so the accuracy shifts a bit based on how the data was altered each time!
plt.title('Privacy-Accuracy Trade-Off')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.show()

hook = syft.TorchHook(torch)

bob = syft.VirtualWorker(hook, id="bob")
alice = syft.VirtualWorker(hook, id="alice")

data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.]], requires_grad=True)
target = torch.tensor([[0], [0], [1], [1.]], requires_grad=True)

# get pointers to training data on each worker by
# sending some training data to bob and alice
data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

# Iniitalize A Toy Model
model = nn.Linear(2, 1)

data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

# organize pointers into a list
datasets = [(data_bob, target_bob), (data_alice, target_alice)]

opt = optim.SGD(params=model.parameters(), lr=0.1)


def train():
    # Training Logic
    opt = optim.SGD(params=model.parameters(), lr=0.1)
    for iter in range(10):

        # NEW) iterate through each worker's dataset separately
        for data, target in datasets:
            # NEW) send model to correct worker - either Alice or Bob
            model.send(data.location)

            # 1) Reset the optimizer so that we can develop a new model
            opt.zero_grad()

            # 2) Predict on new (unseen) data using the model from the cloud
            pred = model(data)

            # 3) See how well (or not) we did on that prediction
            loss = ((pred - target) ** 2).sum()

            # 4) Figure out why we performed poorly
            loss.backward()

            # 5) Update the model's weights
            opt.step()

            # NEW) Get the new model, to be tested and improved on a new, separate dataset
            model.get()

            # 6) print our progress
            print(loss.get())  # NEW) slight edit... need to call .get() on loss\


# federated averaging

train()