import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.output_size = n_classes
        channels, height, width = im_size
        self.input_size = channels*height*width
        self.hidden_size = hidden_dim#32, kernel_size=5
        stride = 2
        padding = 2
        max_pool = 2
        self.output_height = int(1 + (height - kernel_size + 2*padding )/stride)
        self.output_width = int(1 + (width- kernel_size + 2*padding )/stride)
#        self.l1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, dilation=1))
        self.l1 = nn.Sequential(
                nn.Conv2d(channels, hidden_dim, kernel_size, stride=stride, padding=padding, bias=True, dilation=1),
                nn.ReLU(), nn.MaxPool2d(max_pool)
                )
        self.f1 = nn.Linear(int(self.output_height/max_pool)*int(self.output_width/max_pool)*hidden_dim, self.output_size)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        scores = self.l1(images)
        scores = scores.view(scores.size(0), -1)
        scores = self.f1(scores)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

