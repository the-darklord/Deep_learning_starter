import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.output_size = n_classes
#        print(im_size)
        channels, height, width = im_size
        self.input_size = channels*height*width
#        self.W = 0.00001 * np.random.randn(input_size, output_size)
#        self.W = 0.00001 * torch.rand(input_size, output_size)
#        self.b = np.zeros(output_size)
#        self.b = torch.from_numpy(np.zeros(output_size))
        #OR
        self.linear = nn.Linear(self.input_size, self.output_size)
#        print("linear", self.linear)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
#        print(images)# we may have to reshape this tensor 
#        print(images.size())
        modified_images = images.view(-1, self.input_size)
        scores = self.linear(modified_images)
#        scores = np.dot(X, self.W) + self.b
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

