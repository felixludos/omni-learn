
import numpy as np
from ...data import register_dataset, Deviced, Batchable, ImageDataset


@register_dataset('unpaired-translation')
class UnpairedTranslationDataset(ImageDataset):
	def __init__(self, A, dataset1=None, dataset2=None, sel_one=None, swap=None, **kwargs):

		if dataset1 is None:
			dataset1 = A.pull('dataset1')
		if dataset2 is None:
			dataset2 = A.pull('dataset2')

		if sel_one is None:
			sel_one = A.pull('sel-first', True)

		if swap is None:
			swap = A.pull('swap', False)

		if swap:
			dataset1, dataset2 = dataset2, dataset1

		print(f'Sizes: {len(dataset1)} vs {len(dataset2)}')

		super().__init__(A, din=dataset1.din, dout=dataset2.din, **kwargs)

		self.din1, self.dout1 = dataset1.din, dataset1.dout
		self.din2, self.dout2 = dataset2.din, dataset2.dout

		self.dataset1 = dataset1
		self.dataset2 = dataset2

		self.select_first = sel_one

	def __len__(self):
		return len(self.dataset1) * len(self.dataset2)

	def __getitem__(self, item):

		idx1 = item // len(self.dataset2)
		idx2 = item % len(self.dataset2)

		b1 = self.dataset1[idx1]
		b2 = self.dataset2[idx2]

		if self.select_first and isinstance(b1, (list, tuple)):
			b1 = b1[0]
			b2 = b2[0]

		return (b1, b2)


@register_dataset('batched-unpaired-translation')
class BatchedUnpairedTranslationDataset(Batchable, UnpairedTranslationDataset):

	def __getitem__(self, item):

		if isinstance(item, (list, tuple)):
			item = np.array(item)

		idx1 = item // len(self.dataset2)
		idx2 = item % len(self.dataset2)

		b1 = self.dataset1[idx1]
		b2 = self.dataset2[idx2]

		if self.select_first and isinstance(b1, (list, tuple)):
			b1 = b1[0]
			b2 = b2[0]

		return (b1, b2)


# @Dataset('unpaired-translation')
# class UnpairedTranslationDataset(Batchable):
#
# 	def __init__(self, A, dataset1=None, dataset2=None, sel_one=None):
#
# 		mode = A.pull('mode', 'train')
#
# 		if dataset1 is None:
# 			dataset1 = A.pull('dataset1')
# 			if type(dataset1) == dict:
# 				slice1 = dataset1
# 				dataset1 = dataset1[mode]
# 			else:
# 				slice1 = {mode: dataset1}
# 		else:
# 			slice1 = None
# 		if dataset2 is None:
# 			dataset2 = A.pull('dataset2')
# 			if type(dataset2) == dict:
# 				slice2 = dataset2
# 				dataset2 = dataset2[mode]
# 			else:
# 				slice2 = {mode: dataset2}
# 		else:
# 			slice2 = None
#
# 		if sel_one is None:
# 			sel_one = A.pull('sel-first', True)
#
# 		super().__init__(dataset1.din, dataset2.din)
#
# 		self.mode = mode
# 		self._slice1, self._slice2 = slice1, slice2
#
# 		self.din1, self.dout1 = dataset1.din, dataset1.dout
# 		self.din2, self.dout2 = dataset2.din, dataset2.dout
#
# 		self.dataset1 = dataset1
# 		self.dataset2 = dataset2
#
# 		self.select_first = sel_one
#
# 	def __len__(self):
# 		return len(self.dataset1) * len(self.dataset2)
#
# 	def __getitem__(self, item):
#
# 		idx1 = item // len(self.dataset2)
# 		idx2 = item % len(self.dataset2)
#
# 		b1 = self.dataset1[idx1]
# 		b2 = self.dataset2[idx2]
#
# 		if self.select_first and isinstance(b1, (list, tuple)):
# 			b1 = b1[0]
# 			b2 = b2[0]
#
# 		return (b1, b2)
#
# 	def split(self, A):
#
# 		splits = {}
#
# 		cls = self.__class__
#
# 		for key in self._slice1.keys():
# 			if key == self.mode:
# 				splits[key] = self
# 			elif self._slice1[key] is not None and self._slice2[key] is not None:
# 				A.push('mode', key, silent=True)
# 				splits[key] = cls(A, dataset1=self._slice1[key], dataset2=self._slice2[key])
#
# 		return splits
