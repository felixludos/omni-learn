
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset

import torch.nn.functional as F

# Global imports
import os
import sys
import shutil
import time
import numpy as np
import torch.multiprocessing as mp
import h5py as hf

import foundation as fd
import foundation.util as util
from foundation import nets
import foundation.data as data
import pydub
##import torchaudio
#from torchaudio.transforms import MEL2
from mel_spectrograms import MEL_SPEC
import options

def load_mp3(path, normalize=True, cut_zeros=False, threshold=1e-3):
	track = pydub.AudioSegment.from_file(path)
	
	if track.channels > 1:
		track, _ = track.split_to_mono()
	
	if normalize:
		track = pydub.effects.normalize(track)
	
	fs = track.frame_rate
	
	wav = np.frombuffer(track.raw_data, dtype=np.int16) / 2 ** 15
	
	if cut_zeros:
		begin, end = np.where(np.abs(wav) > threshold)[0][[0, -1]]
		wav = wav[begin:end]
		
	return wav, fs

def samples2audio(samples, fs=44100, path=None, normalize=False):

	assert len(samples.shape) == 1
	assert samples.dtype == np.float32
	
	samples = (samples * 2**15).astype(np.int16)
	
	audio = pydub.AudioSegment(data=samples.tobytes(), frame_rate=fs, sample_width=2, channels=1)
	audio = audio.set_channels(2)
	
	if normalize:
		audio = pydub.effects.normalize(audio)
	
	if path is not None:
		audio.export(path, format='mp3')
		print('Audio segment (len={:.2}) saved to {}'.format(audio.duration_seconds, path))
	
	return audio


class MEL_Dataset(Dataset):
	
	def __init__(self, waves, hop=10, ws=50, n_mels=128, ):
		
		self.waves = waves
		
		self.fs = self.waves.fs
		
		
		self.hop = hop * self.fs // 1000
		self.ws = ws * self.fs // 1000
		self.n_mels = n_mels
		
		if hasattr(self.waves, 'seq_len'):
			self.seq_len = self.waves.seq_len
			self.output = (1, self.seq_len // self.hop - (self.ws // self.hop) + 1, self.n_mels)
		
		self.spec = MEL_SPEC(self.fs, ws=self.ws, hop=self.hop, n_mels=self.n_mels)

	def __len__(self):
		
		return len(self.waves)
	
	def __getitem__(self, idx):
		sample = self.waves[idx]
		try:
			x, y = sample
			return self.spec(torch.from_numpy(x).float().unsqueeze(0)), y
		except:
			x, y = sample['x'], sample['y']
			#sample['wav'] = x
			sample['x'] = self.spec(torch.from_numpy(x).float().unsqueeze(0))
			
			return sample
		


class Full_Yt_Dataset(Dataset): # returns full song
	def __init__(self, h5_paths, lbl_name='gid', begin=None, end=None):
		
		self.fs = 44100
		self.lbl_name = lbl_name
		self.paths = np.array(h5_paths)
		assert begin == None and end == None
		
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx):
		with hf.File(self.paths[idx], 'r') as f:
			name = f.attrs['name']
			y = f.attrs[self.lbl_name]
			x = f['wav'].value  # [k:(k + traindata.seq_len)]
			
		return {'x':x, 'y':y, 'name':name, 'path':self.paths[idx]}

class Yt_Dataset(Dataset):
	def __init__(self, h5_paths, lbl_name='gid', seq_len=1000, step_len=1, hop=10, begin=None, end=None):
		self.paths = np.array(h5_paths)
		self.lbl_name = lbl_name
		self.fs = 44100
		
		self.hop = hop * self.fs // 1000
		self.seq_len = seq_len * self.fs // 1000
		assert step_len == 1
		# self.step_len = step_len
		
		self._lens = []
		
		for path in self.paths:
			with hf.File(path, 'r') as f:
				self._lens.append(f['wav'].shape[0])
		
		self._lens = np.array(self._lens)
		
		self.begin = begin
		self.end = end
		if begin is not None:
			self.begin *= self.fs
			self.begin //= 1000
			self._lens -= self.begin
		
		if end is not None:
			self.end *= self.fs
			self.end //= 1000
			self._lens -= self.end
		
		# self._lens //= self.step_len
		self._lens -= self.seq_len
		self._lens //= self.hop
		
		sel = self._lens > 0
		self._lens = self._lens[sel]
		self.paths = self.paths[sel]
		
		self._seq_ind = np.cumsum(self._lens)
		self._len = self._seq_ind[-1]
	
	def __len__(self):
		return self._len
	
	def __getitem__(self, idx):
		
		i = np.searchsorted(self._seq_ind, idx, side='right')
		k = idx - self._seq_ind[i - 1] if i > 0 else idx
		
		k *= self.hop
		
		with hf.File(self.paths[i], 'r') as f:
			y = f.attrs[self.lbl_name]
			x = f['wav'][k:(k + self.seq_len)]
		
		return x, y

class MusicNet_Dataset(Dataset):
	def __init__(self, h5_path, ids=None, seq_len=1000, step_len=1, hop=10, begin=None, end=None):
		self.path = h5_path
		self.fs = 44100
		
		self.hop = hop * self.fs // 1000
		self.seq_len = seq_len * self.fs // 1000
		assert step_len == 1
		#self.step_len = step_len
		
		self._ids = ids
		self._lens = []
		
		with hf.File(self.path, 'r') as f:
			if ids is None:
				self._ids = list(f.keys())
				
			for ID in self._ids:
				self._lens.append(f[ID]['data'].shape[0])
		
		self._lens = np.array(self._lens)
		self._ids = np.array(self._ids)
		
		#print(self._lens, self._ids)
		
		self.begin = begin
		self.end = end
		if begin is not None:
			self.begin *= self.fs
			self.begin //= 1000
			self._lens -= self.begin
			
		if end is not None:
			self.end *= self.fs
			self.end //= 1000
			self._lens -= self.end
			
		#self._lens //= self.step_len
		self._lens -= self.seq_len
		self._lens //= self.hop
		
		sel = self._lens > 0
		self._lens = self._lens[sel]
		self._ids = self._ids[sel]
		
		self._seq_ind = np.cumsum(self._lens)
		self._len = self._seq_ind[-1]
			
	def __len__(self):
		return self._len
	
	def __getitem__(self, idx):
		
		i = np.searchsorted(self._seq_ind, idx, side='right')
		k = idx - self._seq_ind[i - 1] if i > 0 else idx
		
		k *= self.hop
		
		name = self._ids[i]
		
		with hf.File(self.path, 'r') as f:
			y = f[name].attrs['label']
			x = f[name]['data'][k:(k+self.seq_len)]
			
		return x, y


class Full_Song_Yt_Dataset(Dataset):
	def __init__(self, h5_paths, lbl_name='gid', seq_len=1000, step_len=1, begin=0, end=None, batch_size=None):
		self.paths = np.array(h5_paths)
		self.lbl_name = lbl_name
		self.fs = 44100
		
		assert False
		
		self.seq_len = seq_len * self.fs // 1000
		assert step_len == 1
		# self.step_len = step_len
		
		self._lens = []
		
		for path in self.paths:
			with hf.File(path, 'r') as f:
				self._lens.append(f['wav'].shape[0])
		
		self._lens = np.array(self._lens)
		
		self.begin = begin
		self.end = end
		if begin is not None:
			self.begin *= self.fs
			self.begin //= 1000
			self._lens -= self.begin
		
		if end is not None:
			self.end *= self.fs
			self.end //= 1000
			self._lens -= self.end
		
		# self._lens //= self.step_len
		#self._lens -= self.seq_len
		self._lens //= self.seq_len
		self._lens -= 1
		
		self.batches = None
		if batch_size is not None:
			l = self._lens[np.argsort(self._lens)[-batch_size]]
			total = self._lens.sum()
			
			self._lens = self._lens.clip(max=l)
			
			new = self._lens.sum()
			
			print('Lost {:.2f}% of data {}'.format((total-new)/total*100., self._lens.max()))
			
			#print(self._lens[191])
			
			idx = np.arange(len(self._lens))
			self.batches = []
			remaining = self._lens.sum()
			ls = self._lens.copy()
			while remaining > 0:
				picks = min(batch_size, remaining)
				#print(ls.sum())
				self.batches.append(np.argsort(ls)[-picks:])
				ls[self.batches[-1]] -= 1
				remaining = ls.sum()
				#print(np.sort(ls)[-batch_size:])
				
			#print(ls[191])
			
			#print((np.bincount(np.hstack(self.batches))-self._lens).sum())
			
			#print(len(self.batches[-1]), self._lens.sum() % batch_size)
			#quit()
		
			#print(self.batches[:5])
		
			self.batches = np.hstack(self.batches)
		
		sel = self._lens > 0
		self._lens = self._lens[sel]
		self.paths = self.paths[sel]
		
		self._seq_ind = np.cumsum(self._lens)
		self._len = self._seq_ind[-1]
		self.num = len(self._seq_ind)
		
		self._current = np.ones(len(self.paths)).astype(int) * self.begin
	
	def __len__(self):
		return self._len
	
	def __getitem__(self, idx):
		
		if self.batches is None:
			i = np.searchsorted(self._seq_ind, idx, side='right')
		else:
			
			i = self.batches[idx]
		
		
		k = self._current[i]
		
		with hf.File(self.paths[i], 'r') as f:
			y = f.attrs[self.lbl_name]
			x = f['wav'][k:(k + self.seq_len)]
		
		self._current[i] += self.seq_len
		
		return i, x, y
		
	


