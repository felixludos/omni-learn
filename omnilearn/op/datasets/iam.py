
import sys, os
from pathlib import Path

from tqdm import tqdm
import h5py as hf

import omnifig as fig

import xml.etree.ElementTree as ET

from ... import util
# from ..data import Dataset
#
# from ...data import standard_split, Device_Dataset, Info_Dataset, Splitable_Dataset, Testable_Dataset, Batchable_Dataset, Image_Dataset


@fig.Script('format-iam', description='Format IAM handwriting dataset')
def format_iam(A):
	
	root = A.pull('root', '<>path', None)
	if root is None:
		root = util.get_data_dir(A)
	else:
		root = Path(root)
	root = root / 'iam'
	
	lroot = root / 'lines'
	assert lroot.is_dir(), 'Missing IAM Lines dataset (download: https://fki.tic.heia-fr.ch/DBs/iamDB/data/lines.tgz)'
	
	xroot = root / 'xml'
	assert xroot.is_dir(), 'Missing IAM XML meta data (download: https://fki.tic.heia-fr.ch/DBs/iamDB/data/xml.tgz)'
	
	dpath = root / 'lines.h5'
	
	f = hf.File(dpath, 'a+')
	
	if 'text' not in f:
		
		xpaths = list(xroot.glob('**/*.xml'))
		
		
		
		for xpath in tqdm(xpaths):
			tree = ET.parse(xpath)
			root = tree.getroot()
			
			elms = root[1]
			ids = [line.attrib['IDs'] for line in elms]
			lines = [line.attrib['text'] for line in elms]
			
			raise NotImplementedError
		
		pass
	
	
	
	f.close()
	
	pass


