import torch
import numpy as np

class Net:
    def __init__(self, layers_size = [784, 30, 20, 10]):
        self.layers_size = layers_size
        self.n_layers = len(layers_size) - 1
        self.b = [None]
        for layer_size in layers_size[1:]:
            self.b.append(torch.zeros(1, layer_size))

        self.w = [None]
        for previous_layer, actual_layer in zip(layers_size, layers_size[1:]):
            self.w.append(torch.normal(0, 1, size=(previous_layer, actual_layer)))

    def forward(self, a0):
        a0 = a0.view(1, -1)

        assert a0.size() == (1, self.layers_size[0]), f'Error: wrong input size ({a0.size()=})'

        a = [a0]
        z = [None]
        for l in range(1, self.n_layers + 1):
            z.append(torch.mm(a[l-1], self.w[l] + self.b[l]))
            a.append(logistic(z[l]))

            assert z[l].size() == (1, self.layers_size[l])
            assert a[l].size() == (1, self.layers_size[l])

        return a, z
    
    def backward(self, a, z, y):
        delta = [None for _ in range(self.n_layers + 1)]
        grad_w = [None for _ in range(self.n_layers + 1)]

        delta[self.n_layers] = a[self.n_layers] - y
        for l in range(self.n_layers, 1, -1):
            delta[l-1] = torch.mm(delta[l], torch.t(self.w[l])) * logistic_d(z[l-1])

        for l in range(1, self.n_layers + 1):
            grad_w[l] = torch.mm(torch.t(a[l - 1]), delta[l])

        return grad_w, delta

class SGD:
    def __init__(self, model, learning_rate):
        self.learning_rate = learning_rate
        self.model = model

    def step(self, x, y):
        a, z = self.model.forward(x)
        grad_w, grad_b = self.model.backward(a, z, y)

        for l in range(1, self.model.n_layers + 1):
            self.model.w[l] -= self.learning_rate * grad_w[l]
            self.model.b[l] -= self.learning_rate * grad_b[l]

def epoch(x, y, optimiser):
    n = x.size()[0]
    print(f'Train size: {n}')
    for i in torch.randperm(n):
        # print(f'Sample {i} -> Size: {x[i].size()}')
        optimiser.step(x[i], y[i])
    
def train(model, n_epochs, X_train, y_train, X_test, y_test, learning_rate=0.01):
    sgd = SGD(model, learning_rate)
    for i in range(n_epochs):
        print(f'Epoch: {i+1}')
        epoch(X_train, y_train, sgd)
        print(accuracy(X_test, y_test, model))
    

def logistic(x):
    return 1/(1 + torch.exp(-x))

def logistic_d(x):
    return logistic(x)*(1 - logistic(x))

def accuracy(X, y_true, model):
    hits = 0
    n = X.size()[0]
    for i in range(n):
        _, a = model.forward(X[i])
        net_out = a[-1] 
        pred = torch.argmax(net_out)
        if y_true[i] == pred:
            hits += 1
    return hits/n