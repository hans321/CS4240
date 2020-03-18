"""
MIT License

Copyright (c) 2018 Ludovic Trottier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
from scipy import linalg
import theano as th
import theano.tensor as T
import time
from Models import *
from zca import *
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

# https://github.com/devyhia/cifar-10/blob/master/ZCA%20%2B%20Logistic%20Regression.ipynb

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=1e-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = as_float_array(X, copy=self.copy)
        
        self.mean_ = np.mean(X, axis=0)
        
        X = X - self.mean_

        sigma = np.dot(X.T, X) / (X.shape[0] - 1)
        
        U, S, V = np.linalg.svd(sigma)

        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        
        self.components_ = np.dot(tmp, U.T)
        
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed
	
def show(img, rescale=False):
    if rescale:
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save(name, img, rescale=False):
    if rescale:
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)


if __name__ == "__main__":
    train_batch_size = 32
    test_batch_size = 124
    best_loss = float("inf")
    best_epoch = -1
    dataset_path = './cifar10'
    cuda = False 

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    #testset = datasets.CIFAR10(root=dataset_path, train=False, download=True)

    # Display original images
    display_transform = transforms.Compose([
    transforms.ToTensor()
    ])
    loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=display_transform), batch_size=train_batch_size, shuffle=True, **kwargs)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    save("original.png", torchvision.utils.make_grid(images))


    train_mean = trainset.data.mean(axis=(0, 1, 2)) / 255     # [0.49139968  0.48215841  0.44653091]
    train_std = trainset.data.std(axis=(0, 1, 2)) / 255       # [0.24703223  0.24348513  0.26158784]

    transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
    ])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train), batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test), batch_size=test_batch_size, shuffle=False, **kwargs)
    start = time.time()
    trainx=images.numpy().reshape(train_batch_size, 3*32*32)
    whitener = ZCA()
    print(trainx.shape)
    whitener.fit(trainx)
    whitened = whitener.transform(images.reshape(train_batch_size, 3*32*32))
    print(whitened.shape)
    image_tensor = torch.tensor(whitened.reshape(32, 3, 32, 32))
    save("zca.png", torchvision.utils.make_grid(image_tensor), rescale=True)
    print("Duration: ", str(time.time() - start))