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
		
# ZCA and MeanOnlyBNLayer implementations copied from
#   https://github.com/TimSalimans/weight_norm/blob/master/nn.py
#
# Modifications made to MeanOnlyBNLayer:
# - Added configurable momentum.
# - Added 'modify_incoming' flag for weight matrix sharing (not used in this project).
# - Sums and means use float32 datatype.		
		
class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(1) * tensor.size(2) * tensor.size(3) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor[0].size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        batch = tensor.size(0)

        flat_tensor = tensor.view(batch, -1)
        transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string