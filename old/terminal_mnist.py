from NN import *
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn import metrics

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))
y_train = to_categorical(y_train)
x_train = (x_train/255).astype('float32')
x_test = (x_test/255).astype('float32')

net = Network()
net.init([784, 128, 64, 10], ["relu", "tanh", "softmax"], "cross_entropy")

net.fit(x_train, y_train, epochs=5, l_rate=0.01)

y_pred = net.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)

cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)
print(f"Accuracy score: {metrics.accuracy_score(y_test, y_pred):10.5}")
