import matplotlib.pyplot as plt 
import torch
import torchvision

class Net:
    def __init__(self, layers_size = [784, 30, 20, 10]):
        self.n_layers = len(layers_size) - 1
        self.biases = [None]
        for layer_size in layers_size[1:]:
            self.biases.append(torch.zeros(1, layer_size))

        self.weights = [None]
        for previous_layer, actual_layer in zip(layers_size, layers_size[1:]):
            self.weights.append(torch.normal(0, 1, size=(previous_layer, actual_layer)))

    def forward(self, a0):
        a = [a0]
        z = [None]
        for l in range(1, self.n_layers + 1):
            z.append(torch.mm(a[l-1], self.weights[l] + self.biases[l]))
            a.append(logistic(z[l]))

        return z, a


def logistic(x):
    return 1/(1 + torch.exp(-x))

def logistic_d(x):
    return logistic(x)*(1 - logistic(x))

def accuracy(X, y_true):
    print(y_true)
    net = Net()
    _, a = net.forward(X)
    net_out = a[-1]
    n = net_out.size()[0]
    hits = 0
    for i in range(n):
        pred = torch.argmax(net_out[i])
        if y_true[i][pred] == 1:
            hits += 1
    return hits/n


# charge les images et étiquettes pour l'entraînement
train_set = torchvision.datasets.MNIST(root = '.', train = True, download = True)
images_train = train_set.data.view(train_set.data.size()[0], -1) / 255
labels_train = torch.nn.functional.one_hot(train_set.targets)

# pareil pour le test
test_set = torchvision.datasets.MNIST(root = '.', train = False, download = True)
images_test = test_set.data.view(test_set.data.size()[0], -1) / 255
labels_test = test_set.targets

# nombres d'images dans l'ensemble d'apprentissage et celui de test
ntrain = labels_train.size()[0]
ntest = labels_test.size()[0]

print(f'{ntrain=}')
print(f'{ntest=}')

print(accuracy(images_train, labels_train))

# plt.rcParams['figure.constrained_layout.use'] = True
# fig = plt.figure()
# for i in range(12):
#     fig.add_subplot(3, 4, i + 1)
#     plt.imshow(images_train[i])
#     plt.title(f"Un {labels_train[i]}", fontsize=10)
# plt.suptitle("Premières images de l'ensemble d'entraînement")
# plt.show()
