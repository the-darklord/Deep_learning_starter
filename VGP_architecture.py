# -*- coding: utf-8 -*-
"""
Created on Thu Sept 22 21:53:08 2018
For: Starter_kit_Variational_Gaussian_Process_pytorch
Author: Gaurav_Shrivastava 

"""

# Imports
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import pyro.contrib.gp as gp
from torch.distributions.multivariate_normal import MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as kl_div



# Cholesky method for decomposing covariance matrix in
# multivariate gaussian distribution
class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
        # TODO: ideally, this should use some form of solve triangular instead of inverse...
        linv =  l.inverse()
        
        inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        # could re-symmetrise 
        #s = (s+s.t())/2.0
        
        return s



# The VGP implementation from the paper
# "The Variation Gaussian Process " by D.Tran, R. Ranganath, D. Blei

class VGP(nn.Module):
	# Whole architecture
	def __init__(self):
		super(VGP, self).__init__()
		self.omegas = nn.Parameter(torch.ones(100,requires_grad = True))
		self.inputs = nn.Parameter(torch.randn(100,100,requires_grad = True))#,100)
		self.outputs = nn.Parameter(torch.randn(100,50))
		self.outputs.requires_grad = True
		self.sigma_ard = nn.Parameter(torch.ones(1))
		self.sigma_ard.requires_grad = True
		self.fc1 = nn.Linear(784, 400)
		self.fc11 = nn.Linear(809, 400)
		self.fc21 = nn.Linear(400, 100)
		self.fc211 = nn.Linear(400, 50)
		self.fc22 = nn.Linear(400, 100)
		self.fc23 = nn.Linear(400, 100)
		self.fc24 = nn.Linear(400, 50)
		self.fc25 = nn.Linear(400, 50)
		self.fc26 = nn.Linear(400, 1)
		self.fc3 = nn.Linear(25, 400)
		self.fc4 = nn.Linear(400, 784)

	# Encoder Network
	# def encode(self,x):
	# 	h1 = F.relu(self.fc1(x))
	# 	return self.fc21(h1), self.fc211(h1) , self.fc26(h1)


	# Auxilary encoding
	def aux_encode(self,x):
		h11 = F.relu(self.fc11(x))
		return self.fc22(h11),self.fc23(h11),self.fc24(h11),self.fc25(h11)
    
    # Reparametrization trick .Kingma and Welling 2014 'Auto-Encoding Variational Bayes'
	def reparametrize(self, mu, var):
		if self.training:
			eps = torch.randn_like(var)
			return eps.mul(var.pow(2)) + mu
		else:
			return mu

    # Decoder network
	def decode(self,z):
		h2 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h2))


	# Inference for Gaussian Mapping with pseudo inputs
	def inf_q(self,x,inputs,outputs,kernel_params,sigma_ard):
		
		#inputs, outputs,kernel_params = self.encode(x.view(-1,784))
		batch_size = x.shape[0]
		input_dim = inputs.shape[1]
		output_dim = outputs.shape[1]
		kernel_params = kernel_params.view(-1)
		
		# Normalizing inputs
		scaled_input = inputs.mul(kernel_params)
		
		# Initializing latent inputs
		xi = torch.randn(batch_size,input_dim)
		scaled_xi = xi.mul(kernel_params)
		
		# Initializing RBF kernels with automatic relevance determination 
		kernel = gp.kernels.RBF(input_dim=input_dim)
		
		# Finding mappings mean and cov matrices
		Kss = kernel.forward(scaled_input).mul(sigma_ard)
		Kes = kernel.forward(scaled_xi,scaled_input).mul(sigma_ard)
		Kee = kernel.forward(scaled_xi).mul(sigma_ard)
		Kss_inv = torch.inverse(Kss)
		KesKss_inv =torch.matmul(Kes,Kss_inv)
		mean = torch.matmul(KesKss_inv,outputs)
		#print(Kss.shape,Kes.shape,Kee.shape,Kss_inv.shape,KesKss_inv.shape,kernel_params.shape)
		cov =Kee - torch.matmul(KesKss_inv,torch.t(Kes))
		
		# the second term in cov is added for numerical stability of cov
		cov = (cov+0.00000000001)
		L = Cholesky.apply(cov)# torch.potrf(cov,upper = False)
		
		# generate mean of z
		eps = torch.randn(batch_size,output_dim)
		reparam_var_params = torch.matmul(L,eps) + mean
		print(output_dim)
		reparam_mu = reparam_var_params[:,:int(output_dim/2)]
		reparam_sigma = reparam_var_params[:,int(output_dim/2):]
		# generate sigma^2 of z
		# eps = torch.randn(batch_size,output_dim)
		# reparam_var_params_sigma = torch.matmul(L,eps) + mean
		z = self.reparametrize(reparam_mu,reparam_sigma) #
		return mean, cov, z ,reparam_mu, reparam_sigma

	# Draw auxilary inference on the mapping
	def inf_r(self,x,z):
		mlp_inp = torch.cat((x.view(-1,784),z),dim= 1)
		return self.aux_encode(mlp_inp)

	# Compute the forward pass of network
	def forward(self, x): 
		inputs = self.inputs
		outputs = self.outputs
		kernel_params = self.omegas
		sigma_ard = self.sigma_ard

		mean,cov,z,reparam_var_params_mean,reparam_var_params_sigma = self.inf_q(x,inputs,outputs,kernel_params,sigma_ard)

		inputs_mu,inputs_sigma, outputs_mu, outputs_sigma = self.inf_r(x,z)
		x = self.decode(z)
		return x, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma


def loss_function(recon_x,x,mean,cov,inputs_mu,inputs_sigma,outputs_mu,outputs_sigma, reparam_var_params_mean,reparam_var_params_sigma):
	print(recon_x.max())
	# Finding the Reconstruction Error
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

	mu = reparam_var_params_mean
	logvar = (reparam_var_params_sigma.pow(2)+0.000000001).log()
	mu_3 = inputs_mu
	logvar_3 = inputs_sigma
	cov = (cov+0.00000000001)

	# Finding the First KL divergence term 
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	outputs_var = outputs_sigma.exp()
	dim = mean.shape

	# Finding the Second KL divergence term
	KLD_2 = 0
	for  i in range(dim[1]):
		q = MVNormal(mean[:,i],cov)
		r = MVNormal(outputs_mu[:,i],torch.eye(cov.shape[0]).mul(outputs_var[:,i]))
		KLD_2 = KLD_2 + kl_div(q,r)

	# Finding the last log densities given in the loss function
	KLD_3 = -0.5 * torch.sum(1 + logvar_3 - mu_3.pow(2) - logvar_3.exp())

	return BCE - KLD - KLD_2 - KLD_3

def train(epoch, model, optimizer, train_loader, device,args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma = model(data)
        loss = loss_function(recon_batch, data,mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma)
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
            recon_batch, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma = model(data)
            test_loss += loss_function(recon_batch, data, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args['batch_size'], 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results_vgp/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Training settings
args = {}
args['batch_size'] = 100
args['test_batch_size'] = 100
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

model = VGP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(1, args['epochs'] + 1):
	train(epoch, model, optimizer, train_loader, device,args)
	test(epoch, model, optimizer, test_loader, device,args)
	with torch.no_grad():
	    sample = torch.randn(64, 25).to(device)
	    sample = model.decode(sample).cpu()
	    save_image(sample.view(64, 1, 28, 28),
	               'results_vgp/sample_' + str(epoch) + '.png')
