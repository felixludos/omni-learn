import sys, os, time
#%load_ext autoreload
#%autoreload 2
import foundation as fd
import foundation.util as util
import numpy as np
import torch.multiprocessing as mp
import ZODB, ZODB.FileStorage
import ZEO
#import neo
import transaction
import persistent
from persistent.list import PersistentList
from persistent.mapping import PersistentMapping
from BTrees import OIBTree, IOBTree, IIBTree
from html.parser import HTMLParser

from zodb_wikigraph import create_database, WikiGraph_DB
from extraction import wiki_expand, download_article_html

import configargparse

def worker_expand(article_name):
	
	
	links, redirs, _ = wiki_expand(article_name)
	groups = [[link['href'][6:] for link in group] for group in [links, redirs]]
	links, redirs = map(list, map(set, groups))
	if article_name in links:
		links.remove(article_name)
	
	return links, redirs

args = None

def main():
	# len(links), len(redirs)
	parser = configargparse.ArgumentParser(description='Wikipedia Collect Article Links using ZODB')
	
	# parser.add_argument('--seeds', type=str, default=['Artificial_intelligence', 'Computational_physics', 'Control_theory', 'Machine_learning',
	#          'Quantum_computing'], nargs='+')
	
	parser.add_argument('--seeds', type=str,
	                    default=['Artificial_intelligence', ], nargs='+')
	parser.add_argument('--db-path', default='test.zeo', type=str)
	
	
	global args
	args = parser.parse_args()
	
	if not os.path.isfile(args.db_path):
		create_database(args.db_path)
	
	
	addr, stop = ZEO.server(path=args.db_path)
	print('Addr: ip={}, port={}'.format(*addr))
	
	wiki = WikiGraph_DB(addr)
	
	todo = set()
	completed = set()
	
	todo.update(set(args.seeds))
	
	itr = 0
	print_freq = 100
	while len(todo) > 0:
		
		article_name = todo.pop()
		completed.add(article_name)
		
		# print('Out {}'.format(out[0]))
		
		saved = wiki.get_article(wiki.get_ID(article_name)).links if article_name in wiki else None
		if saved is None:
			
			links, redirs = worker_expand(article_name)
			
			todo.update({link: 1 for link in links if (link not in todo and link not in completed)})
			
			wiki.add(article_name, links, redirs)
			
			todo.update([link for link in links if link not in completed])
			
		else:
			print('Skip: {}'.format(article_name))
			
		
		if itr % print_freq == 0:
			print('\n** Itr {}: todo={}, completed={}, db={}, recent={}'.format(itr + 1, len(todo), len(completed), len(wiki), article_name))
		
		# if itr == 10:
		# 	break
		
		itr += 1
	
	print('\nAfter {} iterations - Final size: {}'.format(itr, len(wiki)))
	
	del wiki
	stop()
	
	print('Done.')
	


if __name__ == '__main__':
	main()


	

