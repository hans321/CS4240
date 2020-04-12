# CS4240
TU Delft Deep learning (CS4240)
https://github.com/hans321/CS4240

## Paper
.... [^1]

## Replication
The goal of the replication is to replicate the accuracy of the 12 models listed in Table 3 of the paper.


### Fixing the available code
Investigating the available code lead to the conclusion that there were some differences between the implemented model in the code and the paper's description of the model:

#### Hyperparameters
First we found that the learning rate was set to 0.0001 in the code. The paper states the following set of learning rates was used: 0.25, 0.1, 0.05, 0.01. Unfortunately, the paper doesn't mention which learning rates resulted in the best results for each model but 0.001 as listed in the code definetely isn't one of them. Secondly, the code didn't include any weight-decay while the paper clearly states a weight-decay constant of 0.001 was used. 

#### CIFAR-10 post-processing
The paper description of the post-processing is very clear, they used normalization and image whitening. However, the code only applies normalization with the addition of horizontally flipping and padding. For image whitening the paper refers to a different paper which states the whitening technique used is Zero Component Analysis (ZCA). ZCA essentially makes edges more prominant by filtering out low-order correlations between pixels. ZCA does introduce an additional hyperparameter $\epsilon$ which we fixed to 0.01 in our models. The images below show the effect of ZCA:
![Pre-ZCA](images/original.png)*Original CIFAR-10 images*
![Post-ZCA](images/zca.png)*CIFAR-10 images processed with ZCA*

### Results

Replication results for all models is shown in the table below.

| Model / Learning rate | 0.25 | 0.1   | 0.05  | 0.01  |
|---------------------|------|-------|-------|-------|
| Model A             | x    | 88.05 | 16.74 | 16.17 |
| Strided-CNN-A       | x    | x     | x     | 17.58 |
| ConvPool-CNN-A      | x    | x     | x     | 12.17 |
| ALL-CNN-A           | x    | x     | x     | x     |
| Model B             | x    | x     | x     | 14.06 |
| Strided-CNN-B       | x    | x     | x     | x     |
| ConvPool-CNN-B      | x    | x     | x     | x     |
| ALL-CNN-B           | x    | x     | x     | x     |
| Model C             | x    | x     | x     | 13.28 |
| Strided-CNN-C       | x    | x     | x     | x     |
| ConvPool-CNN-C      | x    | x     | x     | x     |
| ALL-CNN-C           | x    | x     | x     | x     |

The table entries indicate the model error (100 - accuracy).
Entries with an "x" indicate that no useful result could be obtained. We determined that the model was not useful if it was stuck at 90% error for atleast the first 10 epochs while training. This is because from several experimental observations it was observed that a model would never recover after getting stuck at 90% error irrespective of the learning rate.

The following figure illustrates how the train and test error change as function of the amount of epochs. 

![](./images/model_c_0_01.svg)

## Extension

For our extension of the replication we focuessed on two aspects: weight initialization and hyperparameter tunning

### Results

## Hyper parameter tuning
Some discrepancy between the results of the extended version and the results from the paper[^1] can still be observed. In an attempt to somehow reduce this gap we investigated the influence of two hyper parameters parameters: weight-decay and batch size.

For both hyper parameters model ALL-CNN-C was investigated because model C is more explicitly elaborated in the paper[^1] and the ALL-CNN-C was shown to have the best performance of all other models. 

### weight-decay
From the replication results it is clear that the test error is around 10% larger than the train error. Our Hypothesis is that adding regularization will decrease this gap and thus improve the test error. To increase the regularization we attempted to increased the weight-decay.

| Model C ALL CNN | 0.005 | 0.003 | 0.002 | 0.0015 |
|-----------------|-------|-------|-------|--------|
| learning rate = 0.01 | 24.79 | 16.62 | 14.32 | 13.03 |

Increasing the weight-decay did indeed decrease the gap between train and test and often closed it completely. There was however a large negative effect on the train error which caused the final result to still be worse.
The figure below uses a weight-decay of 0.005, eventually resulting in a gap of around 3% points in error between test and error but the test error is significantly larger (11% points) compared to using a weight-decay of 0.001. 

![](./images/model_all_cnn_c_wd_0_005.svg)

Also note that:
Increasing the weight-decay beyond 0.005 appeared to always result in useless models (stuck at 90% error). 
Increasing the learning rate to 0.05 (the next step in the range from the paper[^1]) also resulted in useless models after increasing the weight decay.

Therefore it is assumed the current weight-decay of 0.001 should not be changed.

### batch size

## Conclusion
There is still a significant discrepancy between the replicated results and those shown in the paper[^1]. Several parameters are not mentioned in the paper like batch size and weight initialization. It is not sure if we had known these parameters we would obtain better results than the default ones we used now.

We noticed that in the paper[^1] it is mentioned (in table 1) that a global averaging over 6 x 6 spatial dimensions is performed just before the softmax. After analyzing the models we think this should be a global averaging over 8 x 8 spatial dimensions because otherwise there is a mismatch in dimensional sizes. In the given code we also observed that the averaged array has a size of 8 x 8. This is probably an error in the paper[^1].

Some remaining unclearity originates from the strided model.
For us it is unclear if the strided model uses dropout after the convolution layer wich has an increased stride. In the paper[^1] it is mentioned that "We applied dropout to the input image as well as after each pooling layer (or after the layer replacing the pooling layer respectively)."[^1] but this does not specify what happens to the dropout if the pooling layer is ommited entirely.

Our recommendation to authors is to publish their original code which was used to obtain the results to make sure that all hyperparameters (also those implicitly assumed) are known.


[^1]: https://arxiv.org/abs/1412.6806
