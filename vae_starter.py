# -*- coding: utf-8 -*-
"""
Created on Thu Sept 20 23:43:08 2018
For: Starter_kit_Variational_Autoencoders_pytorch
Author: Gaurav_Shrivastava 

"""

# Imports

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Variational Autoencoder architecture
class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 30)
        self.fc22 = nn.Linear(400, 30)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z):
        h2 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h2))

    def forward(self, x):
        mu,logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu,logvar)
        x = self.decode(z)
        return x, mu, logvar


        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch, model, optimizer, train_loader, device,args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch,model, optimizer, test_loader,device,args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args['batch_size'], 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    # Training settings
    args = {}
    args['batch_size'] = 64
    args['test_batch_size'] = 1000
    args['epochs'] = 10
    args['lr'] = .01
    args['momentum'] = 0.5
    args['seed'] = 1
    args['log_interval'] = 10
    kwargs = {}
    torch.manual_seed(args['seed'])
    device = torch.device("cpu")    
    # Loading Dataset
    train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
                    batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
                    batch_size=args['batch_size'], shuffle=True, **kwargs)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args['epochs'] + 1):
        train(epoch, model, optimizer, train_loader, device,args)
        test(epoch, model, optimizer, test_loader, device,args)
        with torch.no_grad():
            sample = torch.randn(64, 30).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

if __name__ == '__main__':
    main()