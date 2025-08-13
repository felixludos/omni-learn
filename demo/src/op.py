from .imports import *
from omnilearn import *
from omnilearn.op import *
from omnilearn import autoreg
from omnilearn import scripts
from torchvision.datasets import MNIST as Torchvision_MNIST



@fig.component('mnist')
class MNIST(Dataset):
	_val_split = 10000

	split: Literal['train', 'test', 'val'] = hparam('train') # ['train', 'test', 'val']
	download = hparam(True)
	@hparam()
	def dataroot(self) -> Path:
		return my_root / 'data'

	@property
	def size(self):
		split = self.split
		if self._val_split is None:
			assert split in ('train', 'test'), f'Invalid split: {split}'
			size = 60000 if split == 'train' else 10000
		else:
			assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
			assert 0 < self._val_split < 60000, f'Invalid val_split: {self._val_split}'
			size = {'train': 60000-self._val_split, 'val': self._val_split, 'test': 10000}[split]
		return size

	@property
	def name(self) -> str:
		return 'MNIST' if self.split == 'train' else f'MNIST-{self.split}'

	def as_eval(self, **kwargs) -> 'MNIST':
		assert self.split == 'train', 'Only train split can be converted to eval'
		return self.__class__(split='val', **kwargs)

	def setup(self, *, device: str = None):
		if getattr(self, '_dataset', None) is None:
			self._dataset = Torchvision_MNIST(self.dataroot, train=self.split != 'test', download=self.download)
			self._image_data = self._dataset.data
			self._label_data = self._dataset.targets
			if self.split != 'test' and self._val_split is not None:
				if self.split == 'train':
					self._image_data = self._image_data[self._val_split:]
					self._label_data = self._label_data[self._val_split:]
				else:
					self._image_data = self._image_data[:self._val_split]
					self._label_data = self._label_data[:self._val_split]
			if device is not None:
				self._image_data = self._image_data.to(device)
				self._label_data = self._label_data.to(device)
		return super().setup(device=device)

	@tool('image')
	def get_images(self, indices: np.ndarray) -> torch.Tensor:
		'''returns int8 tensor of shape (N, 28, 28)'''
		return self._image_data[torch.from_numpy(indices)]
	@get_images.space
	def image_space(self) -> spaces.Pixels:
		return spaces.Pixels(1, 28, 28, as_bytes=True)

	@tool('label')
	def get_labels(self, indices: np.ndarray) -> torch.Tensor:
		return self._label_data[torch.from_numpy(indices)]
	@get_labels.space
	def label_space(self) -> spaces.Categorical:
		return spaces.Categorical(10)



@fig.component('classification')
class ImageClassification(Machine):
	@tool('observation')
	def transform_image(self, image: torch.Tensor) -> torch.Tensor:
		N, *_ = image.shape
		if image.dtype == torch.uint8:
			image = image.float().div(255)
		return image.view(N, -1)

	@transform_image.space
	def observation_space(self, image: spaces.Pixels) -> spaces.Bounded:
		"""Flattens the image"""
		return spaces.Bounded(image.size, lower=0., upper=1.)

	@space('prediction')
	def prediction_space(self, label: spaces.Categorical) -> spaces.Logits:
		return spaces.Logits(label.n)

	label_space = space('label')

	@indicator('loss')
	def get_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		return F.cross_entropy(prediction, label)

	@tool('correct', space=spaces.Boolean(1))
	def get_correct(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		return (prediction.argmax(dim=1) == label)

	@indicator('accuracy')
	def get_accuracy(self, correct: torch.Tensor) -> torch.Tensor:
		return correct.float().mean()



# @fig.component('nn')
# class NeuralNetwork(Machine, nn.Sequential):
# 	hidden = hparam(None)
# 	nonlin = hparam('elu')
# 	output_nonlin = hparam(None)
#
# 	input_dim = hparam(None)
# 	@space('input')
# 	def input_space(self) -> spaces.Vector:
# 		if self._input_dim is None:
# 			raise self._GearFailed('input_dim is not set')
# 		return spaces.Vector(self._input_dim) if isinstance(self._input_dim, int) \
# 			else self._input_dim
#
# 	output_dim = hparam(None)
# 	@space('output')
# 	def output_space(self) -> spaces.Vector:
# 		if self._output_dim is None:
# 			raise self._GearFailed('output_dim is not set')
# 		return spaces.Vector(self._output_dim) if isinstance(self._output_dim, int) \
# 			else self._output_dim
#
# 	norm = hparam(None)
# 	output_norm = hparam(None)
# 	dropout = hparam(None)
# 	output_dropout = hparam(None)
#
#
# 	# def setup(self, *, device: Optional[str] = None,
# 	# 		  input_space: Optional[spaces.AbstractSpace] = None,
# 	# 		  output_space: Optional[spaces.AbstractSpace] = None):
# 	# 	input_space = input_space or self.input_space
# 	# 	output_space = output_space or self.output_space
# 	# 	layers = self._build(input_space, output_space, hidden=self._hidden,
# 	# 						 nonlin=self._nonlin, output_nonlin=self._output_nonlin,
# 	# 						 norm=self._norm, dropout=self._dropout,
# 	# 						 output_norm=self._output_norm, output_dropout=self._output_dropout)
# 	# 	for i, layer in enumerate(layers):
# 	# 		self.add_module(f'{i}', layer)
# 	#
# 	# 	if device is not None:
# 	# 		self.to(device)



