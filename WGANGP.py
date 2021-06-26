# Implementation of WGAN-GP is based on:
# https://github.com/donand/GAN_pytorch/blob/master/WGAN-GP/wgan_gp.py


import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import load, Loader
import os
import sys
import datetime
import shutil
import pandas as pd
import time
#from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
import cv2 as cv


class Discriminator(nn.Module):
    def __init__(self, input_channels, nf):
        super(Discriminator, self).__init__()
        self.flattened_size = 64 * \
                              (image_size[1] // 2 // 2 // 2) * (image_size[2] // 2 // 2 // 2)
        self.conv_block = nn.Sequential(
            # input is (3, 32, 32)
            nn.Conv2d(input_channels, nf, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf, 16, 16)
            nn.Conv2d(nf, nf * 2, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*2, 8, 8)
            nn.Conv2d(nf * 2, nf * 4, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 8, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # DIFFERENT FILTER
            nn.Conv2d(nf * 8, 1, 4, padding=0, stride=1, bias=False),

        )

    def forward(self, x):
        x = self.conv_block(x)
        return x.view(-1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    def __init__(self, input_size, output_channels, nf=128):
        super(Generator, self).__init__()

        if image_size[1] == 64:
            self.first_block = nn.Sequential(
                nn.ConvTranspose2d(input_size, nf * 8, 4, stride=1,
                                   padding=0, bias=False),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        elif image_size[1] == 128:
            self.first_block = nn.Sequential(
                nn.ConvTranspose2d(input_size, nf * 16, 4, stride=1,
                                   padding=0, bias=False),
                nn.BatchNorm2d(nf * 16),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ConvTranspose2d(nf * 16, nf * 8, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf * 4, nf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf * 2, nf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf, output_channels, 4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.first_block(x)
        x = self.conv_block(x)
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def compute_gradient_penalty(real, fake, discriminator, lambda_pen):
    # Compute the sample as a linear combination
    alpha = torch.rand(real.shape[0], 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)
    x_hat = alpha * real + (1 - alpha) * fake
    # Compute the output
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    out = discriminator(x_hat)
    # compute the gradient relative to the new sample
    gradients = torch.autograd.grad(
        outputs=out,
        inputs=x_hat,
        grad_outputs=torch.ones(out.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # Reshape the gradients to take the norm
    gradients = gradients.view(gradients.shape[0], -1)
    # Compute the gradient penalty
    penalty = (gradients.norm(2, dim=1) - 1) ** 2
    penalty = penalty * lambda_pen
    return penalty


import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(dataroot, nc, image_size, batch_size):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(num_output_channels=nc),  # TRANSFORMS TO GREYSCALE
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))  # change the number of mean, std
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    return dataloader


def initialize_D():
    discriminator = Discriminator(image_size[0], discriminator_filters).to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
    discriminator.weight_init(mean=0.0, std=0.02)

    return discriminator, disc_optimizer


def initialize_G():
    generator = Generator(n_noise_features, image_size[0], generator_filters).to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    generator.weight_init(mean=0.0, std=0.02)

    return generator, gen_optimizer


def train_WGAN(train_loader, epochs, generator, discriminator, disc_optimizer, gen_optimizer):
    img_list = []
    disc_losses, gen_losses, w_distances, gradient_penalty_list = [], [], [], []
    gen_iterations = 0
    steps = 0

    for e in range(epochs):
        if e % print_every == 0:
            print('Epoch {}'.format(e))
        start = time.time()
        epoch_dlosses, epoch_glosses = [], []
        train_iterator = iter(train_loader)
        i = 0
        while i < len(train_loader):
            noise_factor = (epochs - e) / epochs
            #########################
            # Train the discriminator
            #########################
            # for p in discriminator.parameters():  # reset requires_grad
            # p.requires_grad = True
            # train the discriminator disc_steps times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                disc_steps = 100
            else:
                disc_steps = disc_steps_config
            j = 0
            while j < disc_steps and i < len(train_loader):
                j += 1
                i += 1
                images, _ = train_iterator.next()
                images = images.to(device)
                common_batch_size = min(batch_size, images.shape[0])
                disc_optimizer.zero_grad()
                noises = torch.from_numpy(np.random.randn(common_batch_size, n_noise_features)).type(
                    dtype=torch.FloatTensor).to(device)
                # Compute output of both the discriminator and generator
                disc_output = discriminator(images)
                gen_images = generator(noises)
                gen_output = discriminator(gen_images)
                # disc_output.backward(torch.ones(common_batch_size, 1).to(device))
                # gen_output.backward(- torch.ones(common_batch_size, 1).to(device))
                gradient_penalty = compute_gradient_penalty(images, gen_images, discriminator, lambda_pen)
                loss = torch.mean(gen_output - disc_output + gradient_penalty)
                loss.backward()
                wdist = torch.mean(disc_output - gen_output)
                disc_optimizer.step()

                # Save the loss
                # disc_losses.append(torch.mean(errD).item())
                # epoch_dlosses.append(torch.mean(errD).item())
                # writer.add_scalar('data/D_loss', torch.mean(errD).item(), steps)
                disc_losses.append(loss.item())
                epoch_dlosses.append(loss.item())
                w_distances.append(wdist.item())
                gradient_penalty_list.append(torch.mean(gradient_penalty).item())

                steps += 1

            #######################
            # Train the generator
            #######################
            # print('Training generator {} {}'.format(gen_iterations, i))
            # for p in discriminator.parameters():  # reset requires_grad
            # p.requires_grad = False
            gen_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
                dtype=torch.FloatTensor).to(device)
            gen_images = generator(noises)
            gen_output = discriminator(gen_images)
            # gen_output.backward(torch.ones(batch_size, 1).to(device))
            loss = - torch.mean(gen_output)
            loss.backward()
            gen_optimizer.step()
            # Save the loss
            # gen_losses.append(torch.mean(gen_output).item())
            # epoch_glosses.append(torch.mean(gen_output).item())
            # writer.add_scalar('data/G_loss', torch.mean(gen_output).item(), gen_iterations)
            gen_losses.append(loss.item())
            epoch_glosses.append(loss.item())

            # print('------------', gen_loss.item(), np.mean(temp3))
            # print([x.grad for x in list(generator.parameters())])
            gen_iterations += 1

        for x in range(5):
            img = generate_image(n_noise_features)
            status = cv.imwrite('/content/drive/My Drive/WGANGP/WGAN_results/WGAN_generated_e{}_n{}.png'.format(e, x),
                                img * 255)
        if e % print_every == 0:
            # generate_frame(discriminator, generator, e, frame_noise)
            print('D loss: {:.5f}\tG loss: {:.5f}\tTime: {}'.format(
                np.mean(epoch_dlosses), np.mean(epoch_glosses), datetime.now().strftime("%H:%M:%S")))
        # if e % checkpoints == 0:
        # checkpoint(discriminator, generator, e)

    return disc_losses, gen_losses, w_distances, gradient_penalty_list, img_list


def generate_image(n_noise_features):
    noise = torch.from_numpy(np.random.randn(1, n_noise_features)).type(
        dtype=torch.FloatTensor).to(device)
    gen_output = generator(noise).detach()

    img = gen_output.cpu()[0, :, :, :].numpy()
    img = np.transpose(img, (1, 2, 0))
    if grayscale == True:
        img = img[:, :, 0]
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    return img
