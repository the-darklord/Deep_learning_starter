import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import pyro.contrib.gp as gp
from torch.distributions.multivariate_normal import MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as kl_div


class VGP(nn.Module):
	# Whole architecture
	def __init__(self):
		super(VGP, self).__init__()
		self.omegas = torch.ones(10).mul(-1)
		self.sigma_ard = torch.ones(1)
		self.fc1 = nn.Linear(784, 400)
		self.fc11 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 100)
		self.fc211 = nn.Linear(400, 50)
		self.fc22 = nn.Linear(450, 100)
		self.fc23 = nn.Linear(450, 100)
		self.fc24 = nn.Linear(450, 50)
		self.fc25 = nn.Linear(450, 50)
		self.fc26 = nn.Linear(400, 1)
		self.fc3 = nn.Linear(50, 400)
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
		inputs, outputs,kernel_params = self.encode(x.view(-1,784))
		batch_size = inputs.shape[0]
		input_dim = inputs.shape[1]
		output_dim = outputs.shape[1]
		kernel_params = kernel_params.view(-1)
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
		print(Kss.shape,Kes.shape,Kee.shape,Kss_inv.shape,KesKss_inv.shape,kernel_params.shape)
		cov =Kee - torch.matmul(KesKss_inv,torch.t(Kes))
		L = torch.potrf(cov,upper = False)
		# generate mean of z
		eps = torch.randn(batch_size,output_dim)
		reparam_var_params_mean = torch.matmul(L,eps) + mean
		# generate sigma^2 of z
		eps = torch.randn(batch_size,output_dim)
		reparam_var_params_sigma = torch.matmul(L,eps) + mean
		z = reparametrize(reparam_var_params_mean,reparam_var_params_sigma) #
		return mean, cov, z ,reparam_var_params_mean, reparam_var_params_sigma

	def inf_r(self,x,z):
		mlp_inp = torch.concat((x.view(-1,784),z))
		return self.aux_encode(mlp_inp)

	# Compute the forward pass of network
	def forward(self, x): 
		mean,cov,z,reparam_var_params_mean,reparam_var_params_sigma = self.inf_q(x)
		inputs_mu,inputs_sigma, outputs_mu, outputs_sigma = self.inf_r(x,z)
		x = self.decode(z)
		return x, mean, cov, inputs_mu,inputs_sigma, outputs_mu, outputs_sigma,reparam_var_params_mean,reparam_var_params_sigma


def loss(recon_x,x,mean,cov,inputs_mu,inputs_sigma,outputs_mu,outputs_sigma, reparam_var_params_mean,reparam_var_params_sigma):
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

	mu = reparam_var_params_mean
	logvar = reparam_var_params_sigma.log()
	mu_3 = inputs_mu
	logvar_3 = inputs_sigma

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	dim = mean.shape
	KLD_2 = 0
	for  i in range(dim[1]):
		q = MVNormal(mean[:,l],cov)
		r = MVNormal(outputs_mu[:l],torch.eye(cov.shape[0]).mul(outputs_sigma[:l]))
		KLD_2 = KLD_2 + kl_div(q,r)

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
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, args['epochs'] + 1):
    train(epoch, model, optimizer, train_loader, device,args)
    test(epoch, model, optimizer, test_loader, device,args)
    with torch.no_grad():
        sample = torch.randn(64, 30).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results_vgp/sample_' + str(epoch) + '.png')
