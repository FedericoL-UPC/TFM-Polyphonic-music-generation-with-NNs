# The implementation of this RBM is based on:
# https://github.com/GabrielBianconi/pytorch-rbm


import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import os
from PIL import Image


class RBM():

    #   Parameters

    #   n_v             :   Number of visible inputs
    #                       Initialized by 0 but then take value of number of inputs
    #   n_h             :   Number of features want to extract
    #                       Must be set by user
    #   k               :   Sampling steps for contrastive divergance
    #                       Default value is 2 steps
    #   epochs          :   Number of epochs for training RBM
    #                       Must be set by user
    #   mini_batch_size :   Size of mini batch for training
    #                       Must be set by user
    #   alpha           :   Learning rate for updating parameters of RBM
    #                       Default value is 0.001
    #   momentum        :   Reduces large jumps for updating parameters
    #   weight_decay    :   Reduces the value of weight after every step of contrastive divergance
    #   data            :   Data to be fitted for RBM
    #                       Must be given by user or else, thats all useless

    def __init__(self, n_v=0, n_h=0, k=2, epochs=15, mini_batch_size=64, alpha=0.001, momentum=0.9, weight_decay=0.001):
        self.number_features = 0
        self.n_v = n_v
        self.n_h = self.number_features
        self.k = k
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.data = torch.randn(1, device="cuda")

    #   fit            method is called to fit RBM for provided data
    #                  First, data is converted in range of 0-1 cuda float tensors by dividing it by their maximum value
    #                  Here, after calling this method, n_v is reinitialized to number of input values present in data
    #                  number_features must be given by user before calling this method
    #   w              Tensor of weights of RBM
    #                  (n_v x n_h)    Randomly initialized between 0-1
    #   a              Tensor of bias for visible units
    #                  (n_v x 1)      Initialized by 1's
    #   b              Tensor of bias for hidden units
    #                  (n_b x 1)      Initialized by 1's
    #   w_moment       Momentum value for weights
    #                  (n_v x n_h)    Initialized by zeros
    #   a_moment       Momentum values for visible units
    #                  (n_v x 1)      Initialized by zeros
    #   b_moment       Momentum values for hidden units
    #                  (n_h x 1)      Initialized by zeros
    def fit(self):

        self.data /= self.data.max()

        self.data = self.data.type(torch.cuda.FloatTensor)

        self.n_v = len(self.data[0])
        self.n_h = self.number_features

        self.w = torch.randn(self.n_v, self.n_h, device="cuda") * 0.1
        self.a = torch.ones(self.n_v, device="cuda") * 0.5
        self.b = torch.ones(self.n_h, device="cuda")

        self.w_moment = torch.zeros(self.n_v, self.n_h, device="cuda")
        self.a_moment = torch.zeros(self.n_v, device="cuda")
        self.b_moment = torch.zeros(self.n_h, device="cuda")

        errors = self.train()
        return errors

    #   train          This method splits dataset into mini_batch and run for given epoch number of times
    def train(self):
        errors = []
        for epoch_no in range(self.epochs):
            ep_error = 0
            for i in range(0, len(self.data), self.mini_batch_size):
                mini_batch = self.data[i:i + self.mini_batch_size]
                ep_error += self.contrastive_divergence(mini_batch)
            print(datetime.now())
            print('Epoch number: {}'.format(epoch_no))
            # print("Epoch Number  : ", epoch_no, "       Error  : ", ep_error.item())
            errors.append(ep_error.item())  #
        return errors

    #   cont_diverg    It performs contrastive divergance using gibbs sampling algorithm
    #   p_h_0          Value of hidden units for given visivle units
    #   h_0            Activated hidden units as sampled from normal distribution (0 or 1)
    #   g_0            Positive associations of RBM
    #   wv_a           Unactivated hidden units
    #   p_v_h          Probability of hidden neuron to be activated given values of visible neurons
    #   p_h_v          Probability of visible neuron to be activated given values of hidden neurons
    #   p_v_k          Value of visible units for given visivle units after k step Gibbs Sampling
    #   p_h_k          Value of hidden units for given visivle units after k step Gibbs Sampling
    #   g_k            Negative associations of RBM
    #   error          Recontruction error for given mini_batch
    def contrastive_divergence(self, v):
        p_h_0 = self.sample_hidden(v)
        h_0 = (p_h_0 >= torch.rand(self.n_h, device="cuda")).float()
        g_0 = v.transpose(0, 1).mm(h_0)

        wv_a = h_0
        #       Gibbs Sampling step
        for step in range(self.k):
            p_v_h = self.sample_visible(wv_a)
            p_h_v = self.sample_hidden(p_v_h)
            wv_a = (p_h_v >= torch.rand(self.n_h, device="cuda")).float()

        p_v_k = p_v_h
        p_h_k = p_h_v

        g_k = p_v_k.transpose(0, 1).mm(p_h_k)

        self.update_parameters(g_0, g_k, v, p_v_k, p_h_0, p_h_k)

        error = torch.sum((v - p_v_k) ** 2)

        return error

    #   p_v_h     :   Probability of hidden neuron to be activated given values of visible neurons
    #   p_h_v     :   Probability of visible neuron to be activated given values of hidden neurons

    # -----------------------------------Bernoulli-Bernoulli RBM--------------------------------------------
    #   p_h_v    =    sigmoid ( weight  x visible  +  visible_bias )
    #   p_v_h    =    sigmoid (weight.t x hidden   +  hidden_bias )
    # ------------------------------------------------------------------------------------------------------
    def sample_hidden(self, p_v_h):  # Bernoulli-Bernoulli RBM
        wv = p_v_h.mm(self.w)
        wv_a = wv + self.b
        p_h_v = torch.sigmoid(wv_a)
        return p_h_v

    def sample_visible(self, p_h_v):  # Bernoulli-Bernoulli RBM
        wh = p_h_v.mm(self.w.transpose(0, 1))
        wh_b = wh + self.a
        p_v_h = torch.sigmoid(wh_b)
        return p_v_h

    #   weight_(t)        =     weight_(t)       +  ( positive_association  -  negative_association )  +  weight_(t-1)
    #   visible_bias_(t)  =     visible_bias_(t) +  sum( input - activated_visivle_at_k_step_sample )  +  visible_bias_(t-1)
    #   hidden_bias_(t)   =     hidden_bias_(t)  +  sum( activated_initial_hidden - activated_hidden_at_k_step_sample ) + hidden_bias_(t-1)
    def update_parameters(self, g_0, g_k, v, p_v_k, p_h_0, p_h_k):
        self.w_moment *= self.momentum
        del_w = (g_0 - g_k) + self.w_moment

        self.a_moment *= self.momentum
        del_a = torch.sum(v - p_v_k, dim=0) + self.a_moment

        self.b_moment *= self.momentum
        del_b = torch.sum(p_h_0 - p_h_k, dim=0) + self.b_moment

        batch_size = v.size(0)

        self.w += del_w * self.alpha / batch_size
        self.a += del_a * self.alpha / batch_size
        self.b += del_b * self.alpha / batch_size

        self.w -= (self.w * self.weight_decay)

        self.w_moment = del_w
        self.a_moment = del_a
        self.b_moment = del_b


# sample_array = np.reshape(obstructed_image,-1)
# print(sample_array.shape)
def sample_rbm(rbm, input_batch):
    input_batch = [np.reshape(img, -1) for img in input_batch]  # flatten the array
    sample_tensor = torch.tensor(np.array(input_batch), device="cuda")

    p_h_v = rbm.sample_hidden(sample_tensor.float())
    p_v_h = rbm.sample_visible(p_h_v)

    reconstructed_batch = p_v_h.cpu().numpy()

    return reconstructed_batch


def sample_rbm_rand(rbm, rand_batch):
    p_h_v = rbm.sample_hidden(rand_batch.float())
    p_v_h = rbm.sample_visible(p_h_v)
    reconstructed_batch = p_v_h.cpu().numpy()

    return reconstructed_batch


def arrays_to_images(arrays, new_shape):
    reconstructed_images = []
    for array in arrays:
        img = np.reshape(array, new_shape)
        reconstructed_images.append(img)

    return reconstructed_images

