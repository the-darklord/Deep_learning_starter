import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from gpytorch import kernels
import pyro.contrib.gp as gp


class VGP(nn.Module):
    # Whole architecture
    def __init__(self):
        super(VGP, self).__init__()
        self.omegas = torch.ones(10).mul(-1)
        self.sigma_ard = torch.ones(1)
        self.fc1 = nn.Linear(784, 400)
        self.fc11 = nn.Linear(400, 10)
        self.fc21 = nn.Linear(400, 10)
        self.fc211 = nn.Linear(400, 10)
        self.fc22 = nn.Linear(400, 10)
        self.fc23 = nn.Linear(400, 10)
        self.fc24 = nn.Linear(400, 10)
        self.fc25 = nn.Linear(400, 10)
        self.fc26 = nn.Linear(400, 1)
        self.fc3 = nn.Linear(10, 400)
        self.fc4 = nn.Linear(400, 784)

    # Encoder Network
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc211(h1) , self.fc26(h1)

    def aux_encode(self,x):
    	h11 = F.relu(self.fc11(x))
    	return self.fc22(h11),self.fc23(h11),self.fc24(h11),self.fc25(h11)
    
    # Reparametrization trick .Kingma and Welling 2014 'Auto-Encoding Variational Bayes'
    def reparametrize(self, mu, var):
    	logvar = var.log()
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Decoder network
    def decode(self,z):
        h2 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h2))


    def inf_q(self,x):
    	inputs, outputs,kernel_params = encode(x.view(-1,784))
    	batch_size = inputs.shape[0]
    	input_dim = inputs.shape[1]
    	output_dim = outputs.shape[1]
    	scaled_input = inputs.mul(kernel_params)
    	xi = torch.randn(batch_size,input_dim)
    	scaled_xi = xi.mul(kernel_params)
    	kernel = gp.kernels.RBF(input_dim=input_dim)
    	Kss = kernel.forward(scaled_input)
    	Kes = kernel.forward(scaled_xi,scaled_input)
    	Kee = kernel.forward(scaled_xi)
    	Kss_inv = torch.inverse(Kss)
    	KesKss_inv =torch.matmul(Kes,Kss_inv)
    	mean = torch.matmul(KesKss_inv,outputs)
    	cov =Kee - torch.matmul(KesKss_inv,torch.t(Kes))
    	L = torch.potrf(cov,upper = False)
    	# generate mean of z
    	eps = torch.randn(batch_size,output_dim)
    	reparam_var_params_mean = torch.matmul(L,eps) + mean
    	# generate sigma^2 of z
    	eps = torch.randn(batch_size,output_dim)
    	reparam_var_params_sigma = torch.matmul(L,eps) + mean
    	z = reparametrize(reparam_var_params_mean,reparam_var_params_sigma) #
    	return mean, cov, z

    def inf_r(self,x,z):
    	mlp_inp = torch.concat((x.view(-1,784),z))
    	return aux_encode(mlp_inp)

    # Compute the forward pass of network
    def forward(self, x): 
        mean,cov,z = inf_q(x)
        inputs_mu,inputs_sigma, outputs_mu, outputs_sigma = inf_r(x,z)
        x = self.decode(z)
        return x, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma
