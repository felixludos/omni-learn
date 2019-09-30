
import sys, os
import time
import h5py as hf
import numpy as np
import pydub
import torchaudio
from torchaudio import transforms
#import matplotlib.pyplot as plt
import options
from datasets import load_mp3
import torch
import foundation as fd
import foundation.util as util
from foundation.util import Farmer

def init_worker(seed, **other_args):
	np.random.seed(seed)
	torch.manual_seed(seed)
	return {}

def format_mp3(idx, path, save_path, info, cut_zeros=True, normalize=False, **other_args):

	wav, fs = load_mp3(path, normalize=normalize, cut_zeros=cut_zeros)
	
	n = len(wav)

	with hf.File(save_path, 'w') as f:
		f.create_dataset('wav', data=wav)
		for k,v in info.items():
			f.attrs[k] = v
		f.attrs['frame_rate'] = fs

	return idx, save_path, info['mid'], info['gid'], n


def main(args=None):

	if args is None:
		parser = options.setup_format_options()
		args = parser.parse_args()


	assert args.input_dir is not None \
	       or args.input_files is not None \
	       or args.input_path_file is not None, 'no input specified'

	file_paths = []
	cond = lambda f: f[-4:] == '.mp3'
	if args.input_dir is not None:
		file_paths = util.crawl(args.input_dir, cond) if args.recursive \
			else [os.path.join(args.input_dir, name) for name in os.listdir(args.input_dir) if cond(name)]

	elif args.input_files is not None:
		file_paths = args.input_files

	elif args.input_path_file is not None:
		with open(args.input_path_file, 'r') as f:
			file_paths = f.readlines().split('\n')

	file_paths = [name for name in file_paths if cond(name)]
	assert len(file_paths), 'No valid mp3 files found'

	print('Found {} mp3 files'.format(len(file_paths)))
	if len(file_paths) <= 10:
		for path in file_paths:
			print(path)
		print()

	os.makedirs(args.output_dir, exist_ok=True)
	print('Output will be saved into: {}'.format(args.output_dir))


	# full meta: '/home/fleeb/workspace/ml_datasets/audio/yt/meta/full_meta.pth.tar'
	meta = None
	if args.meta_path is not None:
		meta = torch.load(args.meta_path)['tracks']
		print('Meta file {} contains {} records'.format(args.meta_path, len(meta)))

		file_paths = [path for path in file_paths if ''.join(os.path.basename(path).split('.')[:-1]) in meta]
		print('{} input files found in meta'.format(len(file_paths)))

	assert meta is not None, 'must provide a meta file'

	private_args = {
		'cut_zeros': args.cut_zeros,
		'normalize': args.normalize,
	}
	print(private_args)
	save_template = os.path.join(args.output_dir, args.output_template)

	def itr_files():
		for idx, path in enumerate(file_paths):
			name = ''.join(os.path.basename(path).split('.')[:-1])
			#name = os.path.basename(path) # includes extension
			rec = meta[name]
			yield {'idx':idx,
			       'path':path,
			       'info':{'mid':rec['mood_id'], 'gid':rec['genre_id'], 'name':name},
			       'save_path': save_template.format(str(idx).zfill(1+int(np.log10(len(file_paths))))),
			}

	processor = Farmer(format_mp3, private_args=private_args, init_fn=init_worker,
	                 unique_worker_args=[{'seed': s} for s in range(args.num_workers)],
	                 volatile_gen=itr_files(), num_workers=args.num_workers, waiting=len(file_paths))

	print_freq = max(1, len(file_paths) // 100)
	count = 0

	with open(os.path.join(args.output_dir, 'meta.csv'), 'w') as f:
		for idx, path, mid, gid, n in processor:

			track = os.path.basename(path)
			f.write('{},{},{},{}\n'.format(track, mid, gid, n))

			if count % print_freq == 0:
				print('{}/{} complete'.format(count+1, len(file_paths)))

			count += 1

if __name__ == '__main__':
	main()