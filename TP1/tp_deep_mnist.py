import matplotlib.pyplot as plt 
import torch
import torchvision

from network import Net, train


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

print(images_train.size())

train(Net(), 10, images_train, labels_train, images_test, labels_test)

# plt.rcParams['figure.constrained_layout.use'] = True
# fig = plt.figure()
# for i in range(12):
#     fig.add_subplot(3, 4, i + 1)
#     plt.imshow(images_train[i])
#     plt.title(f"Un {labels_train[i]}", fontsize=10)
# plt.suptitle("Premières images de l'ensemble d'entraînement")
# plt.show()
