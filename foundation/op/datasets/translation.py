

from ..data import Dataset

from ...data import standard_split, Device_Dataset, Info_Dataset, Splitable_Dataset, \
	Testable_Dataset, Batchable_Dataset, Image_Dataset


@Dataset('unpaired-translation')
class UnpairedTranslationDataset(Batchable_Dataset, Info_Dataset):

	def __init__(self, A, dataset1=None, dataset2=None, sel_one=None):

		mode = A.pull('mode', 'train')

		if dataset1 is None:
			dataset1 = A.pull('dataset1')
			if type(dataset1) == dict:
				dataset1 = dataset1[mode]
		if dataset2 is None:
			dataset2 = A.pull('dataset2')
			if type(dataset2) == dict:
				dataset2 = dataset2[mode]

		if sel_one is None:
			sel_one = A.pull('sel-first', True)

		super().__init__(dataset1.din, dataset2.din)

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

	def split(self, A):

		return None

		ratio = A.pull('val_split', None)

		# if ratio is not None:



		raise NotImplementedError

