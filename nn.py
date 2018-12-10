import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def calculate_loss(model, X, y):
    loss = 0.0
    for i in range(len(X)):
        a = X[i].reshape(1, 2) @ model["W1"] + model["b1"]
        h = np.tanh(a)
        z = h @ model["W2"] + model["b2"]
        y_hat = softmax(z)
        #print("y_hat is ", y_hat )
        if y[i] == 0:
            #print("adding ", y_hat[0][0])
            loss += np.log(y_hat[0][0])
        else:
            loss += np.log(y_hat[0][1])
    return loss * (-1/len(X))




def predict(model, x):
    a = x @ model["W1"] + model["b1"]
    h = np.tanh(a)
    z = h @ model["W2"] + model["b2"]
    y_hat = softmax(z)
    list = []
    for item in y_hat:
        list.append(np.argmax(item))
    return np.array(list)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    eta = 0.01
    W1 = np.random.rand(2, nn_hdim)
    b1 = np.random.rand(1, nn_hdim)
    W2 = np.random.rand(nn_hdim, 2)
    b2 = np.random.rand(1, 2)
    model = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }
    for j in range(1, num_passes + 1):
        for i in range(len(X)):
            # print(y[i])
            #print(model["b1"].shape)
            a = X[i].reshape(1, 2) @ model["W1"] + model["b1"]
            h = np.tanh(a)
            z = h @ model["W2"] + model["b2"]
            y_hat = softmax(z)
            # print(y_hat)
            # print( "")
            if y[i] == 1:
                dLdy_hat = y_hat - np.array([0, 1])
            else:
                dLdy_hat = y_hat - np.array([1, 0])

            dLda = (1 - (np.tanh(a) ** 2)) * (dLdy_hat @ model["W2"].transpose())
            dLdW2 = h.transpose() @ dLdy_hat
            dLdb2 = dLdy_hat
            dLdW1 = X[i].reshape(1, 2).T @ dLda
            dLdb1 = dLda
            model["b1"] = model["b1"] - eta * (dLdb1)
            model["b2"] = model["b2"] - eta * (dLdb2)
            model["W1"] = model["W1"] - eta * (dLdW1)
            model["W2"] = model["W2"] - eta * (dLdW2)
        if print_loss and (j % 1000) == 0 and j != 0:
            print(nn_hdim, "loss is " ,calculate_loss(model, X, y))
    return model

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

#model = build_model(X, y, 1, 100)

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim, 20000, True)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
