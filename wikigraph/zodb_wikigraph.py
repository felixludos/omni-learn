

import sys, os, time
#%load_ext autoreload
#%autoreload 2
import foundation as fd
import foundation.util as util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import h5py as hf
from itertools import product

import ZODB, ZODB.FileStorage
import ZEO
#import neo
import transaction
import persistent
from persistent.list import PersistentList
from BTrees import OIBTree, IOBTree, IIBTree


def create_database(path):
	if os.path.isfile(path):
		raise Exception('File {} already exists, overwriting is not permitted'.format(path))

	path_dir = os.path.dirname(path)
	if len(path_dir) and not os.path.isdir(path_dir):
		os.makedirs(path_dir)
		print('Created dir: {}'.format(path_dir))

	addr, stop = ZEO.server(path=path)

	db = ZEO.DB(addr)
	connection = db.open()
	root = connection.root

	# top level objects:
	# - ID table (name -> ID)
	# - ID list (ID -> name)
	# - rID table (redir -> rID)
	# - rID list (rID -> redir)
	# - num list (ID -> num)
	# - redirect list (ID -> redirects)
	# - link list (ID -> links)

	root.ID_table = OIBTree.BTree()
	root.rID_table = OIBTree.BTree()
	root.article_list = PersistentList()
	root.redir_list = PersistentList()

	transaction.commit()

	db.close()
	stop()

class Article(persistent.Persistent):
	def __init__(self, ID, name, links=None, redirs=None):
		self.ID = ID
		self.name = name
		self.num = 1

		self.links = links
		if links is not None:
			self.add_links(links)

		self.redirs = redirs
		if redirs is not None:
			self.add_redirs(redirs)

	def __len__(self):
		return len(self.links)
	
	def __repr__(self):
		return 'Article[{}]({}, n={}, l={})'.format(self.ID, self.name, self.num, len(self.links) if self.links is not None else '[]')

	def add_links(self, links):
		if self.links is None:
			self.links = IOBTree.Set()
		self.links.update(links)

	def add_redirs(self, redirs):
		if self.redirs is None:
			self.redirs = IOBTree.Set()
		self.redirs.update(redirs)

	def increment(self):
		self.num += 1


class WikiGraph_DB(object): # no multiprocessing with this class (yet)
	def __init__(self, addr):
		self.db = ZEO.DB(addr)
		self.conn = self.db.open()
		self.root = self.conn.root

	def __len__(self):
		return len(self.root.ID_table)

	def _add_article(self, name):
		
		ID = self.root.ID_table.get(name, None)
		
		if ID is None:
			ID = len(self.root.ID_table)
			self.root.ID_table[name] = ID
			self.root.article_list.append(Article(ID, name))
			
		return self.root.article_list[ID]
	
	def _add_IDs(self, names):

		IDs = []

		for name in names:
			#print(name)
			ID = self.root.ID_table.get(name, None)
			if ID is None:
				ID = len(self.root.ID_table)
				self.root.ID_table[name] = ID
				self.root.article_list.append(Article(ID, name))
			else:
				self.root.article_list[ID].num += 1

			IDs.append(ID)
		#
		# if len(names) == 1:
		# 	return IDs[0]

		return IDs
	
	def _add_rIDs(self, names):

		IDs = []

		for name in names:
			#print(name)
			ID = self.root.rID_table.get(name, None)
			if ID is None:
				ID = len(self.root.ID_table)
				self.root.rID_table[name] = ID
				self.root.redir_list.append(name)

			IDs.append(ID)

		# if len(names) == 1:
		# 	return IDs[0]

		return IDs
	
	def add(self, name, links, redirs):
		
		article = self._add_article(name)
		
		if article.links is not None:
			print('Short circuit')
			transaction.abort()
			return
		
		links = self._add_IDs(links)
		redirs = self._add_rIDs(redirs)
		
		article.add_links(links)
		article.add_redirs(redirs)
		
		transaction.commit()

	def _increment(self, *IDs):
		for ID in IDs:
			self.root.num_list[ID] += 1

	def increment(self, *IDs):
		self._increment(*IDs)
		transaction.commit()

	def __contains__(self, name):
		return name in self.root.ID_table

	def get_ID(self, *names):
		out = [self.root.ID_table[name] for name in names]
		if len(names) == 1:
			out = out[0]
		return out
	
	def get_rID(self, *names):
		out = [self.root.rID_table[name] for name in names]
		if len(names) == 1:
			out = out[0]
		return out

	def get_article(self, *IDs):
		out = [self.root.article_list[ID] for ID in IDs]
		if len(IDs) == 1:
			out = out[0]
		return out

	def get_redirs(self, *rIDs):
		out = [self.root.redir_list[rID] for rID in rIDs]
		if len(rIDs) == 1:
			out = out[0]
		return out


	def __del__(self):
		self.conn.close()










#
# def oldcreate_database(path):
# 	if os.path.isfile(path):
# 		raise Exception('File {} already exists, overwriting is not permitted'.format(path))
#
# 	path_dir = os.path.dirname(path)
# 	if len(path_dir) and not os.path.isdir(path_dir):
# 		os.makedirs(path_dir)
# 		print('Created dir: {}'.format(path_dir))
#
# 	addr, stop = ZEO.server(path=path)
#
# 	db = ZEO.DB(addr)
# 	connection = db.open()
# 	root = connection.root
#
# 	# top level objects:
# 	# - ID table (name -> ID)
# 	# - ID list (ID -> name)
# 	# - rID table (redir -> rID)
# 	# - rID list (rID -> redir)
# 	# - num list (ID -> num)
# 	# - redirect list (ID -> redirects)
# 	# - link list (ID -> links)
#
# 	root.ID_table = OIBTree.BTree()
# 	root.rID_table = OIBTree.BTree()
# 	root.link_list = PersistentList()
# 	root.redir_list = PersistentList()
# 	root.num_list = PersistentList()
# 	root.article_list = PersistentList()
# 	root.rID_list = PersistentList()
#
# 	transaction.commit()
#
# 	db.close()
# 	stop()

# class OldWikiGraph_DB(object): # no multiprocessing with this class (yet)
# 	def __init__(self, addr):
# 		self.db = ZEO.DB(addr)
# 		self.conn = self.db.open()
# 		self.root = self.conn.root
#
# 	def _add_article(self, *names):
#
# 		IDs = []
#
# 		home = self.root.ID_table
#
# 		created = False
#
# 		for name in names:
# 			ID = home.get(name, None)
# 			if ID is None:
# 				created = True
# 				ID = len(home)
# 				home[name] = ID
# 				self.root.link_list.append(IOBTree.Set())
# 				self.root.redir_list.append(IOBTree.Set())
# 				self.root.num_list.append(1)
# 				self.root.article_list.append(name)
# 			else:
# 				self.root.num_list[ID] += 1
#
# 			IDs.append(ID)
#
# 		if len(names) == 1:
# 			return IDs[0], created
#
# 		return IDs, created
#
# 	def add_article(self, *names):
# 		out = self._add_article(*names)[0]
# 		transaction.commit()
# 		return out
#
# 	def _add_redir(self, *redirs):
#
# 		IDs = []
#
# 		home = self.root.rID_table
#
# 		created = False
#
# 		for name in redirs:
# 			ID = home.get(name, None)
# 			if ID is None:
# 				created = True
# 				ID = len(home)
# 				home[name] = ID
# 				self.root.rID_list.append(name)
#
# 			IDs.append(ID)
#
# 		if len(redirs) == 1:
# 			return IDs[0]
#
# 		return IDs, created
#
# 	def add_redir(self, *redirs):
# 		out = self._add_redir(*redirs)
# 		transaction.commit()
# 		return out
#
# 	def add(self, name, links, redirs):
# 		ID, is_new = self._add_article(name)
#
# 		if not is_new:
# 			self.root.num_list[ID] -= 1
# 			transaction.commit()
# 			print('Short circuit')
# 			return
#
# 		# get IDs
# 		links = self._add_article(*links)[0] # increments links automatically
# 		redirs = self._add_redir(*redirs)[0]
#
# 		for link in links:
# 			self.root.link_list[ID].add(link)
#
# 		for redir in redirs:
# 			self.root.redir_list[ID].add(redir)
#
# 		transaction.commit()
#
# 	def _increment(self, *IDs):
# 		for ID in IDs:
# 			self.root.num_list[ID] += 1
#
# 	def increment(self, *IDs):
# 		self._increment(*IDs)
# 		transaction.commit()
#
# 	def get_ID(self, *names):
# 		out = [self.root.ID_table[name] for name in names]
# 		if len(names) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_rID(self, *names):
# 		out = [self.root.rID_table[name] for name in names]
# 		if len(names) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_article_name(self, *IDs):
# 		out = [self.root.article_list[ID] for ID in IDs]
# 		if len(IDs) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_redir_name(self, *IDs):
# 		out = [self.root.rID_list[ID] for ID in IDs]
# 		if len(IDs) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_num(self, *IDs):
# 		out = [self.root.num_list[ID] for ID in IDs]
# 		if len(IDs) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_links(self, *IDs):
# 		out = [self.root.link_list[ID] for ID in IDs]
# 		if len(IDs) == 1:
# 			out = out[0]
# 		return out
#
# 	def get_redirs(self, *IDs):
# 		out = [self.root.redir_list[ID] for ID in IDs]
# 		if len(IDs) == 1:
# 			out = out[0]
# 		return out
#
#
# 	def __del__(self):
# 		self.conn.close()
