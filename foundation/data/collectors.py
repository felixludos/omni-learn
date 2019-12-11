
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py as hf

from .. import util

class DatasetWrapper(util.Simple_Child, Dataset):
	def __init__(self, dataset):
		super().__init__(_parent=dataset)
		self.dataset = dataset


# class Movable_Dataset(Dataset):
#
# 	def __init__(self, dataset, device=None):
# 		self.dataset = dataset
#
# 		self.device = device
#
# 	def __len__(self):
# 		return len(self.dataset)
#
# 	def __getitem__(self, idx):
#
# 		out = self.dataset[idx]
#
# 		if not isinstance(out, Movable):
# 			if isinstance(out, torch.Tensor):
# 				new = out
# 			elif isinstance(out, (list, tuple)):
# 				new = TensorList(out)
# 			elif isinstance(out, dict):
# 				new = TensorDict()
# 				new.update(out)
# 		else:
# 			raise Exception('Unknown type {}: {}'.format(type(out), out))
#
# 		if self.device is not None:
# 			new = new.to(self.device)
#
# 		return new

class Batchable_Dataset(Dataset): # you can select using a full batch
	def allow_batched(self):
		return True

class Info_Dataset(Dataset):

	def __init__(self, din, dout, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.din = din
		self.dout = dout

	def get_info(self):
		return self.din, self.dout

	def pre_epoch(self, mode, epoch): # TODO: integrate in train.running
		pass

	def post_epoch(self, mode, epoch, stats=None):
		pass

# class Resizeable_Dataset(Dataset):
# 	def __init__(self, *args, size=None, **kwargs):
# 		super().__init__(*args, **kwargs)

class Device_Dataset(Dataset): # Full dataset is in memory, so it can be moved to GPU
	def __init__(self, *args, device='cpu', **kwargs):
		super().__init__(*args, **kwargs)
		self.device = device
		self._buffers = set()

	def register_buffer(self, name, buffer):
		self._buffers.add(name)
		self.__setattr__(name, buffer)

	def get_device(self):
		return self.device

	def to(self, device):
		self.device = device
		for name in self._buffers:
			try:
				val = getattr(self,name)
				if val is not None:
					self.__setattr__(name, val.to(device))
			except AttributeError:
				pass

class Testable_Dataset(Dataset): # Has a pre-defined loadable testset
	def __init__(self, *args, train=True, **kwargs):
		super().__init__(*args, **kwargs)
		self.train = train

	def get_mode(self):
		return 'train' if self.train else 'test'

class Indexed_Dataset(Dataset):
	def __init__(self, d):
		self.dataset = d

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return idx, self.dataset[idx]

class Replay_Buffer(Dataset):
	def __init__(self, buffer_size):
		assert False, 'deprecated, see rl/rlhw_backend.py'
		self.buffer_size = buffer_size
		self.reset()
		
	def reset(self):
		self.size = 0
		self.ptr = 0
		self.buffer = [None] * self.buffer_size
		
	def add(self, sample):
		self.buffer[self.ptr] = sample
		self.ptr += 1
		self.ptr %= len(self.buffer)
		self.size = min(self.size+1, len(self.buffer))
		
	def __getitem__(self, idx):
		return self.buffer[idx]
	
	def __len__(self):
		return self.size

class List_Dataset(Dataset):

	def __init__(self, ls):
		self.data = ls

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

class Subset_Dataset(DatasetWrapper):

	def __init__(self, dataset, indices):
		super().__init__(dataset)
		self.indices = indices

		try:
			device = self.dataset.get_device()
			self.indices = self.indices.to(device)
		except AttributeError:
			pass

	def __getitem__(self, idx):
		return self.dataset[self.indices[idx]]

	def __len__(self):
		return len(self.indices)

class Format_Dataset(DatasetWrapper):

	def __init__(self, dataset, format_fn, format_args=None, include_original=False):
		super().__init__(dataset)

		self.format_fn = format_fn
		self.format_args = {} if format_args is None else format_args
		self.include_original = include_original

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):

		sample = self.dataset[idx]

		formatted = self.format_fn(sample, **self.format_args)

		if self.include_original:
			return formatted, sample

		return formatted

class Shuffle_Dataset(DatasetWrapper):

	def __init__(self, dataset):
		super().__init__(dataset)

		self.indices = torch.randperm(len(self.dataset))

		try:
			device = self.dataset.get_device()
			self.indices = self.indices.to(device)
		except AttributeError:
			pass

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[self.indices[idx]]

class Uneven_Seq_Dataset(Dataset):

	def __init__(self, dataset, seq_lens): # seq_lens: length of sequence for each sample in dataset

		assert len(dataset) == len(seq_lens)
		self.seq_lens = np.array(seq_lens)
		self.seq_ind = self.seq_lens.cumsum()
		self.dataset = dataset

	def __len__(self):
		return self.seq_ind[-1]

	def __getitem__(self, idx):

		i = np.searchsorted(self.seq_ind, idx, side='right')
		k = idx - self.seq_ind[i-1] if i > 0 else idx

		#print(idx, i, k, self.seq_ind[i-1], self.seq_lens[i])

		return k, self.dataset[i]

class Seq_Dataset(Dataset):

	def __init__(self, dataset, num_seq):

		self.dataset = dataset
		self.num_seq = num_seq # number of sequences in a single sample in the dataset

	def __len__(self):
		return len(self.dataset) * self.num_seq

	def __getitem__(self, idx):

		b = idx // self.num_seq
		k = idx % self.num_seq

		#print(idx, b, k)

		return k, self.dataset[b]

class H5_Dataset(Dataset):

	def __init__(self, h5_file_path, keys=None, unbatched_keys=None):

		self.path = h5_file_path

		self.unbatched = unbatched_keys

		with hf.File(self.path, 'r') as h5_file:

			if keys is None:
				keys = {k for k in h5_file.keys()}
			self.keys = set((key if isinstance(key, tuple) else (key,)) for key in keys)

			if self.unbatched is not None: # TODO: add select_grouped to unbatched keys
				for k in self.unbatched:
					assert k in h5_file, '{} not found'.format(k)

			# check first dim of all data arrays is equal
			lens = np.array([self.select_grouped(h5_file, *key).shape[0] for key in self.keys])
			assert (lens == lens[0]).all(), 'not all lengths are the same for these keys'
			self._len = lens[0]

	def select_grouped(self, obj, *terms):
		for term in terms:
			obj = obj[term]
		return obj

	def __getitem__(self, idx):

		with hf.File(self.path, 'r') as h5_file:
			sample = {'-'.join(key):self.select_grouped(h5_file, *key)[idx] for key in self.keys}
			if self.unbatched is not None:
				sample.update({k:h5_file[k].value for k in self.unbatched})
		return sample

	def __len__(self):
		return self._len

class H5_Flattened_Dataset(Dataset):

	def __init__(self, h5_file_path, keys=None, unbatched_keys=None):

		self.path = h5_file_path

		self.unbatched = unbatched_keys

		with hf.File(self.path, 'r') as h5_file:

			if keys is None:
				keys = {k for k in h5_file.keys()}
			self.keys = keys

			if self.unbatched is not None:
				for k in self.unbatched:
					assert k in h5_file, '{} not found'.format(k)

			# check first dim of all data arrays is equal
			lens = np.array([h5_file[k].shape[0] for k in self.keys])
			self._seq = np.array([h5_file[k].shape[1] for k in self.keys]).min()
			assert (lens == lens[0]).all(), 'not all lengths are the same for these keys'
			self._len = lens[0]


	def __getitem__(self, idx):

		i = idx // self._len
		j = idx % self._seq

		with hf.File(self.path, 'r') as h5_file:
			sample = {k:h5_file[k][i,j] for k in self.keys}
			if self.unbatched is not None:
				sample.update({k:h5_file[k].value for k in self.unbatched})
		return sample

	def __len__(self):
		return self._len * self._seq


class H5_Loader_Dataset(Dataset): # index refers to a full h5 file, instead of the first dim of data in the file
	def __init__(self, h5_filenames, keys=None):

		self.paths = h5_filenames

		if keys is None:

			with hf.File(self.paths[0], 'r') as h5_file:
				keys = {k for k in h5_file.keys()}

		self.keys = keys

	def __getitem__(self, idx):

		with hf.File(self.paths[idx], 'r') as h5_file:
			sample = {k:h5_file[k].value for k in self.keys}
		return sample

	def __len__(self):
		return len(self.paths)




class Npy_Loader_Dataset(Dataset):

	def __init__(self, npy_file_path, keys=None, unbatched_keys=None):

		self.path = npy_file_path

		self.unbatched_keys = unbatched_keys



		np_file = np.load(npy_file_path)

		if keys is None:
			keys = {k for k in np_file.keys()}
		self.keys = keys

		self.data = {k:np_file[k] for k in self.keys}
		self.unbatched = {k:np_file[k] for k in self.unbatched_keys} if self.unbatched_keys is not None else {}

		# check first dim of all data arrays is equal
		lens = np.array([v.shape[0] for v in self.data.values()])
		assert (lens == lens[0]).all(), 'not all lengths are the same for these keys'
		self._len = lens[0]

	def __getitem__(self, idx):

		sample = {k:self.data[k][idx] for k in self.keys}
		sample.update(self.unbatched)
		return sample

	def __len__(self):
		return self._len

