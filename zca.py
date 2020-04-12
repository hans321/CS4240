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
try:
    import cupy as cp
except ModuleNotFoundError:
    #!pip install cupy
    import cupy as cp

import cupy as cp
import numpy as np
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
from scipy import linalg
import pickle as pkl
import time
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import Tensor
import matplotlib.pyplot as plt

# https://github.com/devyhia/cifar-10/blob/master/ZCA%20%2B%20Logistic%20Regression.ipynb

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=1e-1, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        print("Fitting ... ", end="")

        X = as_float_array(X, copy=self.copy)
        self.mean_ =  cp.mean(X, axis=0)
        X = X - self.mean_
        sigma = cp.dot(X.T, X) / (X.shape[0] - 1)

        U, S, V = linalg.svd(sigma)
        tmp = cp.dot(cp.array(U), cp.diag(1 / cp.sqrt(cp.array(S) + self.regularization)))
        self.components_ = cp.dot(tmp, cp.array(U).T)

        print("done")
        return self

    def transform(self, X):
        #print("Transforming ... ", end="")
        X_transformed = X - self.mean_
        X_transformed = cp.dot(cp.array(X_transformed), self.components_.T)

        #print("done")
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
    train_batch_size = 50000
    test_batch_size = 124
    best_loss = float("inf")
    best_epoch = -1
    dataset_path = './cifar10'
    cuda = False 

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR10(root=dataset_path, train=False, download=True)

    # Get mean and std of training set
    train_mean = trainset.data.mean(axis=(0, 1, 2)) / 255     # [0.49139968  0.48215841  0.44653091]
    train_std = trainset.data.std(axis=(0, 1, 2)) / 255       # [0.24703223  0.24348513  0.26158784]

    # Configure transformations
    transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(train_mean, train_std),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(train_mean, train_std),
    ])

    # Load CIFAR10
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train), batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test), batch_size=test_batch_size, shuffle=False, **kwargs)
   
    start = time.time()

    # Load images of train loader
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    train_zca = images.numpy().reshape(train_batch_size, 3*32*32)
    print("Input size:", str(train_zca.shape))

    # Apply ZCA whitening
    zca = ZCA()
    zca.fit(train_zca)
    cp.cuda.Stream.null.synchronize()
    trainset_zca = zca.transform(images.reshape(train_batch_size, 3*32*32))  
    cp.cuda.Stream.null.synchronize()

    print("Duration: ", str(time.time() - start))

    print("Result component matrix:", str(zca.components_.shape))
    print("Result mean matrix:", str(zca.mean_.shape))

    images = images[0:2]
    images_zca = trainset_zca[0:2].reshape(2, 3, 32, 32)
    save("original.png", torchvision.utils.make_grid(torch.tensor(images)), rescale=True)
    save("zca.png", torchvision.utils.make_grid(torch.tensor(images_zca)), rescale=True)    