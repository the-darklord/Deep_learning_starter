import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        #############################################################################
        #  Initialize anything you need for the forward pass
        #############################################################################
        self.output_size = n_classes
        channels, height, width = im_size
        self.input_size = channels*height*width
        self.hidden_size = hidden_dim

        # defining the layers
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu_layer = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, self.output_size)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        #  Implement the forward pass. This should take very few lines of code.
        #############################################################################
        modified_images = images.view(-1, self.input_size)
        scores = self.l1(modified_images)
        scores = self.relu_layer(scores)
        scores = self.l2(scores)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

