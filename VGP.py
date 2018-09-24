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

from gpytorch import kernels







# Variational Autoencoder architecture
class VGP(nn.Module):
    # Whole architecture
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 30)
        self.fc22 = nn.Linear(400, 30)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 784)

    # Encoder Network
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)
    
    # Reparametrization trick .Kingma and Welling 2014 'Auto-Encoding Variational Bayes'
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
               return eps.mul(std).add_(mu)
        else:
            return mu

    def generating_variational_data(self,mu,logvar,size = 100):
        # multiply mu and log var to proper signs
        data = self.reparametrize(self,mu,logvar)
        return data.split(list(mu.shape)[0],dim = 1)#split data into input output pairs

    def gp_mapping_prior(self,inputs,parameters):
        omegas, sigma_ard = parameters.split(list(parameters.shape)[0]-1) 
        ard_kernel = kernels.RBFKernel(ard_num_dims = list(omegas.shape)[0])
        return ard_kernel.forward(inputs.mul(omegas.float())).mul(sigma_ard), omegas, sigma_ard

    # def generate_xi(self,dims):
    #     xi = ?
    #     return xi

    def gp_mapping_pred(self, inputs, output, omegas, sigma_ard, Kss):
        Kss = self.gp_mapping_prior(inputs,theta)
        dim = inputs.shape
        Kss = Kss.view(Kss.shape[1],-1)
        batch_size = inputs.shape[0]
        # get \xi values
        xi = torch.randn(batch_size,1,dim[1])
        ard_kernel = kernels.RBFKernel(ard_num_dims = dim[1])
        Kes = ard_kernel.forward(xi.mul(omegas.float()),inputs.mul(omegas.float())).mul(sigma_ard)
        Kss_inv = torch.inverse(Kss)
        Kee = ard_kernel.forward(xi.mul(omegas.float())).mul(sigma_ard)
        mean = torch.matmul(torch.matmul(Kes,Kss_inv),outputs)
        mean = mean.view(mean.shape[0],-1)
        #cholsky decomposition
        cov = Kee - torch.matmul(torch.matmul(Kes,Kss_inv),Kes.view(Kes.shape[0],Kes.shape[2],-1))
        L = torch.sqrt(cov)
        L = torch.ones(mean.shape).mul(L).view(mean.shape[0],-1)
  
        return mean, cov

    def generate_z(self, mu, cov):
        if self.training:
            L = torch.sqrt(cov)
            L = torch.ones(mean.shape).mul(L).view(mean.shape[0],-1)
            eps = torch.randn_like(mu)
            return eps.mul(L).add_(mu)
        else:
            return mu

    # Decoder network
    def decode(self,z):
        h2 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h2))

    # Compute the forward pass of network
    def forward(self, x):     
        theta,phi = self.encode(x.view(-1,1,784)) # specially for mnist
        inputs,outputs = self.generating_variational_data(),#split phi)
        mean, cov = self.gp_mapping_pred(inputs,outputs,theta)#,Kss)
        z = self.generate_z(mean,cov)
        x = self.decode(z)
        return x, mean, cov


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x,z, mu_1,var_1,mu_2, logvar_2,mu_3,logvar_3):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    eps = torch.randn_like(z)
    KLD_1 = F.kl_div(z, eps)
    
    KLD_2 = -0.5*torch.sum(1-logvar_2+var_1.log() - (var_1+(mu_2-mu_1).pow(2)).div(logvar_2.exp()))
    KLD_3 = -0.5 * torch.sum(1 - logvar_3 - mu_3.pow(2).logvar_3.exp())
    
    # D Tran. Variational Gaussian Process. ICLR, 2016
    return BCE + KLD_1 + KLD_2 + KLD_3

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

def test(epoch,model, optimizer, test_loader,device,args): #batch_test = batch_train
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