from .imports import *



class Mish(nn.Module):
	def forward(self, x):
		return x * torch.tanh(F.softplus(x))
class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)



def get_nonlinearity(ident, dim=1, inplace=True, **kwargs):
	if ident is None:
		return None
	if not isinstance(ident, str):
		return ident

	if ident == 'prelu':
		return nn.PReLU(**kwargs)
	elif ident == 'lrelu':
		return nn.LeakyReLU(**kwargs)
	elif ident == 'relu':
		return nn.ReLU(inplace=inplace)
	elif ident == 'tanh':
		return nn.Tanh()
	elif ident == 'log-softmax':
		return nn.LogSoftmax(dim=dim)
	elif ident == 'softmax':
		return nn.Softmax(dim=dim)
	elif ident == 'softmax2d':
		return nn.Softmax2d()
	elif ident == 'softplus':
		return nn.Softplus(**kwargs)
	elif ident == 'sigmoid':
		return nn.Sigmoid()
	elif ident == 'elu':
		return nn.ELU(inplace=inplace, **kwargs)
	elif ident == 'selu':
		return nn.SELU(inplace=inplace, **kwargs)

	elif ident == 'mish':
		return Mish()
	elif ident == 'swish':
		return Swish()

	else:
		assert False, f'Unknown nonlinearity: {ident}'


