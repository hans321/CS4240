import torch.nn as nn
import torch.nn.functional as F

#---------------------------------------------------------------------------#
#                               A MODELS                                    #
#---------------------------------------------------------------------------#

class Model_A(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_A, self).__init__()                             # Input 32 x 32 RGB image
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5 x 5 conv. 96 ReLU
        
        self.pool2 = nn.MaxPool2d(3, padding=1, stride=2)           # 3 x 3 max-pooling stride 2
        
        self.conv3 = nn.Conv2d(96, 192, 5, padding=2)               # 5 x 5 conv. 192 ReLU
        
        self.pool4 = nn.MaxPool2d(3, padding=1, stride=2)           # 3 x 3 max-pooling stride 2
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)              # 3 x 3 conv. 192 ReLU
        
        self.conv6 = nn.Conv2d(192, 192, 1)                         # 1 x 1 conv. 192 ReLU
        
        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1 x 1 conv. 10 ReLU
        
                                                                    # global averaging over 6 x 6 spatial dimensions
        
        print("Loaded A-Base")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Input 32 x 32 RGB image
        
        conv1_out = F.relu(self.conv1(x_drop))              # 5 x 5 conv. 96 ReLU
        
        pool2_out = self.pool2(conv1_out)                   # 3 x 3 max-pooling stride 2
        pool2_out_drop = F.dropout(pool2_out, .5)
        
        conv3_out = F.relu(self.conv3(pool2_out_drop))      # 5 x 5 conv. 192 ReLU
             
        pool4_out = self.pool4(conv3_out)                   # 3 x 3 max-pooling stride 2
        pool4_out_drop = F.dropout(pool4_out, .5)
		
        conv5_out = F.relu(self.conv5(pool4_out_drop))      # 3 x 3 conv. 192 ReLU
        
        conv6_out = F.relu(self.conv6(conv5_out))           # 1 x 1 conv. 192 ReLU

        class_out = F.relu(self.class_conv(conv6_out))      # 1 x 1 conv. 10 ReLU
        
        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # global averaging over 6 x 6 spatial dimensions
        
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class Model_A_Strided(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_A_Strided, self).__init__()                         # Input 32 x 32 RGB image
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2, stride=2)  # 5 x 5 conv. 96 ReLU stride 2
        
        self.conv2 = nn.Conv2d(96, 192, 5, padding=2, stride=2)         # 5 x 5 conv. 192 ReLU stride 2
        
        self.conv3 = nn.Conv2d(192, 192, 3, padding=1)                  # 3 x 3 conv. 192 ReLU
        
        self.conv4 = nn.Conv2d(192, 192, 1)                             # 1 x 1 conv. 192 ReLU
        
        self.class_conv = nn.Conv2d(192, n_classes, 1)                  # 1 x 1 conv. 10 ReLU

                                                                        # global averaging over 6 x 6 spatial dimensions

        print("Loaded A-Strided")

        # Note: no dropout after convolution layers?
    def forward(self, x):
        x_drop = F.dropout(x, .2)                               # Input 32 x 32 RGB image
        
        conv1_out = F.relu(self.conv1(x_drop))                  # 5 x 5 conv. 96 ReLU
        
        conv2_out = F.relu(self.conv2(conv1_out))               # 5 x 5 conv. 192 ReLU
        	
        conv3_out = F.relu(self.conv3(conv2_out))               # 3 x 3 conv. 192 ReLU
        
        conv4_out = F.relu(self.conv4(conv3_out))               # 1 x 1 conv. 192 ReLU
        
        class_out = F.relu(self.class_conv(conv4_out))          # 1 x 1 conv. 10 ReLU
        
        pool_out = F.adaptive_avg_pool2d(class_out, 1)          # global averaging over 6 x 6 spatial dimensions
        
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

		
class Model_A_ConvPool(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_A_ConvPool, self).__init__()                # Input 32 x 32 RGB image

        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)    # 5 x 5 conv. 96 ReLU
        self.conv2 = nn.Conv2d(96, 96, 5, padding=2)            # 5 x 5 conv. 96 ReLU

        self.pool3 = nn.MaxPool2d(3, padding=1, stride=2)       # 3 x 3 max-pooling stride 2

        self.conv4 = nn.Conv2d(96, 192, 5, padding=2)           # 5 x 5 conv. 192 ReLU
        self.conv5 = nn.Conv2d(192, 192, 5, padding=2)          # 5 x 5 conv. 192 ReLU

        self.pool6 = nn.MaxPool2d(3, padding=1, stride=2)       # 3 x 3 max-pooling stride 2
        
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)          # 3 x 3 conv. 192 ReLU
        
        self.conv8 = nn.Conv2d(192, 192, 1)                     # 1 x 1 conv. 192 ReLU

        self.class_conv = nn.Conv2d(192, n_classes, 1)          # 1 x 1 conv. 10 ReLU

                                                                # global averaging over 6 x 6 spatial dimensions

        print("Loaded A-ConvPool")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                               # Input 32 x 32 RGB image
        
        conv1_out = F.relu(self.conv1(x_drop))                  # 5 x 5 conv. 96 ReLU
        conv2_out = F.relu(self.conv2(conv1_out))               # 5 x 5 conv. 96 ReLU
        
        pool3_out = self.pool3(conv2_out)                       # 3 x 3 max-pooling stride 2
        pool3_out_drop = F.dropout(pool3_out, .5)

        conv4_out = F.relu(self.conv4(pool3_out_drop))          # 5 x 5 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))               # 5 x 5 conv. 192 ReLU
					
        pool6_out = self.pool6(conv5_out)                       # 3 x 3 max-pooling stride 2
        pool6_out_drop = F.dropout(pool6_out, .5)

        conv7_out = F.relu(self.conv7(pool6_out_drop))          # 3 x 3 conv. 192 ReLU
        
        conv8_out = F.relu(self.conv8(conv7_out))               # 1 x 1 conv. 192 ReLU

        class_out = F.relu(self.class_conv(conv8_out))          # 1 x 1 conv. 10 ReLU
        
        pool_out = F.adaptive_avg_pool2d(class_out, 1)          # global averaging over 6 x 6 spatial dimensions
        
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
		
class Model_A_All(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_A_All, self).__init__()                         # Input 32 x 32 RGB image

        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5 x 5 conv. 96 ReLU

        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, stride=2)      # 3 x 3 max-pooling stride 2

        self.conv3 = nn.Conv2d(96, 192, 5, padding=2)               # 5 x 5 conv. 192 ReLU

        self.conv4 = nn.Conv2d(192, 192, 3, padding=1, stride=2)    # 3 x 3 max-pooling stride 2
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)              # 3 x 3 conv. 192 ReLU
        
        self.conv6 = nn.Conv2d(192, 192, 1)                         # 1 x 1 conv. 192 ReLU

        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1 x 1 conv. 10 ReLU

                                                                    # global averaging over 6 x 6 spatial dimensions

        print("Loaded A-All")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Input 32 x 32 RGB image
        
        conv1_out = F.relu(self.conv1(x_drop))              # 5 x 5 conv. 96 ReLU

        conv2_out = F.relu(self.conv2(conv1_out))           # 3 x 3 max-pooling stride 2
        conv2_out_drop = F.dropout(conv2_out, .5)

        conv3_out = F.relu(self.conv3(conv2_out_drop))      # 5 x 5 conv. 192 ReLU

        conv4_out = F.relu(self.conv4(conv3_out))           # 3 x 3 max-pooling stride 2
        conv4_out_drop = F.dropout(conv4_out, .5)
		
        conv5_out = F.relu(self.conv5(conv4_out_drop))      # 3 x 3 conv. 192 ReLU
        
        conv6_out = F.relu(self.conv6(conv5_out))           # 1 x 1 conv. 192 ReLU

        class_out = F.relu(self.class_conv(conv6_out))      # 1 x 1 conv. 10 ReLU
        
        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # global averaging over 6 x 6 spatial dimensions
        
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

#---------------------------------------------------------------------------#
#                               B MODELS                                    #
#---------------------------------------------------------------------------#
class Model_B(nn.Module):
    """ Bade model B 
    """

    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_B, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5x5 conv. 96 ReLU
        self.conv2 = nn.Conv2d(96, 96, 1)                           # 1x1 conv. 96 ReLU
        
        self.max1 = nn.MaxPool2d(3, padding=1, stride=2)            # 3x3 max-pooling stride 2
        
        self.conv3 = nn.Conv2d(96, 192, 5, padding=2)               # 5x5 conv. 192 ReLU
        self.conv4 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU 

        self.max2 = nn.MaxPool2d(3, padding=1, stride=2)            # 3x3 max-pooling stride 2
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)              # 3x3 conv. 192 ReLU
        self.conv6 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU

        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1x1 conv. 10 ReLU

        print("Loaded model B")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Dropout is 20% for all inputs (see end of p. 5)

        conv1_out = F.relu(self.conv1(x_drop))              # 5x5 conv. 96 ReLU             
        conv2_out = F.relu(self.conv2(conv1_out))           # 1x1 conv. 96 ReLU
        
        max1_out = self.max1(conv2_out)                     # 3x3 max-pooling stride 2
        max1_out_drop = F.dropout(max1_out, .5)             # Dropout is 50% for all other than inputs (see end of p. 5)

        conv3_out = F.relu(self.conv3(max1_out_drop))       # 5x5 conv. 192 ReLU
        conv4_out = F.relu(self.conv4(conv3_out))           # 1x1 conv. 192 ReLU
        
        max2_out = self.max2(conv4_out)                      # 3x3 max-pooling stride 2
        max2_out_drop = F.dropout(max2_out, .5)             # Dropout is 50% for all other than inputs (see end of p. 5)
        
        conv5_out = F.relu(self.conv5(max2_out_drop))       # 3x3 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))           # 1x1 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv6_out))      # 1x1 conv. 10 ReLU

        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # Global averaging pooling with output size of 1
        
        pool_out.squeeze_(-1)   
        pool_out.squeeze_(-1)   
        return pool_out

class Strided_CNN_B(nn.Module):
    """ Max-pooling is removed and the stride of the conv. layers preceeding 
        the max-pool layers is increased by 1
    """
    
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Strided_CNN_B, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5x5 conv. 96 ReLU
        self.conv2 = nn.Conv2d(96, 96, 1, stride=2)                 # 1x1 conv. 96 ReLU
                
        self.conv3 = nn.Conv2d(96, 192, 5, padding=2)               # 5x5 conv. 192 ReLU
        self.conv4 = nn.Conv2d(192, 192, 1, stride=2)               # 1x1 conv. 192 ReLU 
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)              # 3x3 conv. 192 ReLU
        self.conv6 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU

        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1x1 conv. 10 ReLU

        print("Loaded Strided-CNN-B")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Dropout is 20% for all inputs (see end of p. 5)

        conv1_out = F.relu(self.conv1(x_drop))              # 5x5 conv. 96 ReLU                   
        conv2_out = F.relu(self.conv2(conv1_out))           # 1x1 conv. 96 ReLU
        #conv2_out_drop = F.dropout(conv2_out, .5)           # Dropout is 50% for all other than inputs (see end of p. 5)
        
        conv3_out = F.relu(self.conv3(conv2_out))           # 5x5 conv. 192 ReLU
        conv4_out = F.relu(self.conv4(conv3_out))           # 1x1 conv. 192 ReLU 
        #conv4_out_drop = F.dropout(conv4_out, .5)           # Dropout is 50% for all other than inputs (see end of p. 5)
        
        conv5_out = F.relu(self.conv5(conv4_out))      # 3x3 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))           # 1x1 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv6_out))      # 1x1 conv. 10 ReLU

        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # Global averaging pooling with output size of 1
        
        pool_out.squeeze_(-1)   
        pool_out.squeeze_(-1)   
        return pool_out

class ConvPool_CNN_B(nn.Module):
    """ A dense convolution is placed before each max-pooling layer
    """

    def __init__(self, input_size, n_classes=10, **kwargs):
        super(ConvPool_CNN_B, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5x5 conv. 96 ReLU
        self.conv2 = nn.Conv2d(96, 96, 1)                           # 1x1 conv. 96 ReLU
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)                # 3x3 conv. 96 ReLU
 
        self.max1 = nn.MaxPool2d(3, stride=2)                       # 3x3 max-pooling stride 2
        
        self.conv4 = nn.Conv2d(96, 192, 5, padding=2)               # 5x5 conv. 192 ReLU
        self.conv5 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)              # 3x3 conv. 192 ReLU

        self.max2 = nn.MaxPool2d(3, stride=2)                       # 3x3 max-pooling stride 2
        
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)              # 3x3 conv. 192 ReLU
        self.conv8 = nn.Conv2d(192, 192, 1)                         # 3x3 conv. 192 ReLU
        self.conv9 = nn.Conv2d(96, 96, 3, padding=1)                # 3x3 conv. 96 ReLU

        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1x1 conv. 10 ReLU

        print("Loaded ConvPool-CNN-B")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Dropout is 20% for all inputs (see end of p. 5)

        conv1_out = F.relu(self.conv1(x_drop))              # 5x5 conv. 96 ReLU                   
        conv2_out = F.relu(self.conv2(conv1_out))           # 1x1 conv. 96 ReLU
        conv3_out = F.relu(self.conv3(conv2_out))           # 3x3 conv. 96 ReLU

        max1_out = self.max1(conv3_out)                     # 3x3 max-pooling
        conv3_out_drop = F.dropout(max1_out, .5)            # Dropout is 50% for all other than inputs (see end of p. 5)

        conv4_out = F.relu(self.conv4(conv3_out_drop))      # 5x5 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))           # 1x1 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))           # 3x3 conv. 192 ReLU

        max2_out = self.max2(conv6_out)                     # 3x3 max-pooling
        conv6_out_drop = F.dropout(max2_out, .5)            # Dropout is 50% for all other than inputs (see end of p. 5)
        
        conv7_out = F.relu(self.conv7(conv6_out_drop))      # 3x3 conv. 192 ReLU
        conv8_out = F.relu(self.conv8(conv7_out))           # 3x3 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv8_out))      # 3x3 conv. 96 ReLU

        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # Global averaging pooling with output size of 1
        
        pool_out.squeeze_(-1)   
        pool_out.squeeze_(-1)   
        return pool_out

class All_CNN_B(nn.Module):
    """ Max-pooling is replaced by a conv. layer.
    """
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(All_CNN_B, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, 96, 5, padding=2)        # 5x5 conv. 96 ReLU
        self.conv2 = nn.Conv2d(96, 96, 1)                           # 1x1 conv. 96 ReLU
        
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)      # 3x3 conv. 96 ReLU
        
        self.conv4 = nn.Conv2d(96, 192, 5, padding=2)               # 5x5 conv. 192 ReLU
        self.conv5 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU 

        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)    # 3x3 conv. 192 ReLU
        
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)              # 3x3 conv. 192 ReLU
        self.conv8 = nn.Conv2d(192, 192, 1)                         # 1x1 conv. 192 ReLU
        self.class_conv = nn.Conv2d(192, n_classes, 1)              # 1x1 conv. 10 ReLU

        print("All-CNN-B")

    def forward(self, x):
        x_drop = F.dropout(x, .2)                           # Dropout is 20% for all inputs (see end of p. 5)

        conv1_out = F.relu(self.conv1(x_drop))              # 5x5 conv. 96 ReLU                  
        conv2_out = F.relu(self.conv2(conv1_out))           # 1x1 conv. 96 ReLU

        conv3_out = F.relu(self.conv3(conv2_out))           # 3x3 conv. 96 ReLU
        conv3_out_drop = F.dropout(conv3_out, .5)           # Dropout is 50% for all other than inputs (see end of p. 5)

        conv4_out = F.relu(self.conv4(conv3_out_drop))      # 5x5 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))           # 1x1 conv. 192 ReLU 

        conv6_out = F.relu(self.conv6(conv5_out))           # 3x3 conv. 192 ReLU
        conv6_out_drop = F.dropout(conv6_out, .5)           # Dropout is 50% for all other than inputs (see end of p. 5)
        
        conv7_out = F.relu(self.conv7(conv6_out_drop))      # 3x3 conv. 192 ReLU
        conv8_out = F.relu(self.conv8(conv7_out))           # 1x1 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv8_out))      # 1x1 conv. 10 ReLU

        pool_out = F.adaptive_avg_pool2d(class_out, 1)      # Global averaging pooling with output size of 1
        
        pool_out.squeeze_(-1)   
        pool_out.squeeze_(-1)   
        return pool_out

#---------------------------------------------------------------------------#
#                               C MODELS                                    #
#---------------------------------------------------------------------------#

class Model_C(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Model_C, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)        
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)

        print("Loaded Model C")


    def forward(self, x):
        x_drop = F.dropout(x, .2)                       # Dropout is 20% for all inputs (see end of p. 5)
        conv1_out = F.relu(self.conv1(x_drop))          # 3x3 conv. 96 ReLU
        conv2_out = F.relu(self.conv2(conv1_out))       # 3x3 conv. 96 ReLU
        pool1_out = F.relu(self.pool1(conv2_out))       # 3x3 max-pooling stride 2
        pool1_out_drop = F.dropout(pool1_out, .5)       # Dropout 50% after every max-pooling layer
        conv4_out = F.relu(self.conv4(pool1_out_drop))  # 3x3 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))       # 3x3 conv. 192 ReLU
        pool2_out = F.relu(self.pool2(conv5_out))       # 3x3 max-pooling stride 2
        pool2_out_drop = F.dropout(pool2_out, .5)       # Dropout 50% after every max-pooling layer
        conv7_out = F.relu(self.conv7(pool2_out_drop))  # 3x3 conv. 192 ReLU
        conv8_out = F.relu(self.conv8(conv7_out))       # 1x1 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv8_out))  # 1x1 conv. 10 ReLU
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # Global averaging
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class Strided_CNN_C(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(Strided_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)        
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)

        print("Loaded Strided-CNN-C")


    def forward(self, x):
        x_drop = F.dropout(x, .2)                       # Dropout is 20% for all inputs (see end of p. 5)
        conv1_out = F.relu(self.conv1(x_drop))          # 3x3 conv. 96 ReLU
        conv2_out = F.relu(self.conv2(conv1_out))       # 3x3 conv. 96 ReLU with stride 2
        conv3_out = F.relu(self.conv3(conv2_out))       # 3x3 conv. 192 ReLU
        conv4_out = F.relu(self.conv4(conv3_out))       # 3x3 conv. 192 ReLU with stride 2
        conv5_out = F.relu(self.conv5(conv4_out))       # 3x3 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))       # 1x1 conv. 192 ReLU
        class_out = F.relu(self.class_conv(conv6_out))  # 1x1 conv. 10 ReLU
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # Global averaging
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class ConvPool_CNN_C(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(ConvPool_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)

        print("Loaded ConvPool-C")


    def forward(self, x):
        x_drop = F.dropout(x, .2)                       # Dropout is 20% for all inputs (see end of p. 5)
        conv1_out = F.relu(self.conv1(x_drop))          # 3x3 conv. 96 ReLU
        conv2_out = F.relu(self.conv2(conv1_out))       # 3x3 conv. 96 ReLU
        conv3_out = F.relu(self.conv3(conv2_out))       # 3x3 conv. 96 ReLU
        
        pool1_out = self.pool1(conv3_out)               # 3x3 max-pooling stride 2
        pool1_out_drop = F.dropout(pool1_out, .5)       # Dropout 50% after every max pool layer

        conv4_out = F.relu(self.conv4(pool1_out_drop))  # 3x3 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))       # 3x3 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))       # 3x3 conv. 192 ReLU

        pool2_out = self.pool2(conv6_out)               # 3x3 max-pooling stride 2
        pool2_out_drop = F.dropout(pool2_out, .5)       # Dropout 50% after every max pool layer

        conv7_out = F.relu(self.conv7(pool2_out_drop))  # 3x3 conv. 192 ReLU
        conv8_out = F.relu(self.conv8(conv7_out))       # 1x1 conv. 192 ReLU

        class_out = F.relu(self.class_conv(conv8_out))  # 1x1 conv. 10 ReLU
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # Global averaging 
        pool_out.squeeze_(-1)                           
        pool_out.squeeze_(-1)
        return pool_out

class ALL_CNN_C(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(ALL_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)

        print("Loaded ALL-CNN-C")


    def forward(self, x):
        x_drop = F.dropout(x, .2)                       # Dropout is 20% for all inputs (see end of p. 5)
        conv1_out = F.relu(self.conv1(x_drop))          # 3x3 conv. 96 ReLU
        conv2_out = F.relu(self.conv2(conv1_out))       # 3x3 conv. 96 ReLU
        conv3_out = F.relu(self.conv3(conv2_out))       # 3x3 conv. 96 ReLU with stride 2
        conv3_out_drop = F.dropout(conv3_out, .5)       # Dropout 50% after layer replacing max-pooling (see end of p. 5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))  # 3x3 conv. 192 ReLU
        conv5_out = F.relu(self.conv5(conv4_out))       # 3x3 conv. 192 ReLU
        conv6_out = F.relu(self.conv6(conv5_out))       # 3x3 conv. 192 ReLU with stride 2
        conv6_out_drop = F.dropout(conv6_out, .5)       # Dropout 50% after layer replacing max-pooling (see end of p. 5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))  # 3x3 conv. 192 ReLU
        conv8_out = F.relu(self.conv8(conv7_out))       # 1x1 conv. 192 ReLU

        class_out = F.relu(self.class_conv(conv8_out))  # 1x1 conv. 10 ReLU
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # Global averaging
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
