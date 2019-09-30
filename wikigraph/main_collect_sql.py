import sys, os, time
#%load_ext autoreload
#%autoreload 2
import foundation as fd
import foundation.util as util
import numpy as np
import itertools
import urllib
from urllib.error import URLError
import http
from http.client import IncompleteRead
import torch.multiprocessing as mp
from html.parser import HTMLParser

from sql_wikigraph import create_database, WikiGraph_DB
from extraction import wiki_expand, download_article_html

import configargparse

def worker_expand(article_name):

	try:

		links, redirs, _, b = wiki_expand(article_name)
		groups = [[link['href'][6:] for link in group] for group in [links, redirs]]
		links, redirs = map(set, groups)
		if article_name in links:
			links.remove(article_name)

	except (URLError, IncompleteRead) as e:

		return article_name, None
	
	return article_name, links, redirs, b

args = None

def main():
	# len(links), len(redirs)
	parser = configargparse.ArgumentParser(description='Wikipedia Collect Article Links using ZODB')
	
	# parser.add_argument('--seeds', type=str, default=['Artificial_intelligence', 'Computational_physics', 'Control_theory', 'Machine_learning',
	#          'Quantum_computing'], nargs='+')
	
	parser.add_argument('--seeds', type=str,
	                    default=['Artificial_intelligence', ], nargs='+')
	parser.add_argument('--db-path', default='test.zeo', type=str)
	parser.add_argument('-j','--num-workers', default=4, type=int)
	parser.add_argument('--batch-size', default=16, type=int)
	
	
	global args
	args = parser.parse_args()
	
	if not os.path.isfile(args.db_path):
		create_database(args.db_path)

	print('Database path: {}'.format(args.db_path))

	failed_file = open(os.path.join(os.path.dirname(args.db_path), 'failed.txt'), 'r')
	failed = set(failed_file.readlines())
	failed_file.close()
	failed_file = open(os.path.join(os.path.dirname(args.db_path), 'failed.txt'), 'a+')

	print('Already failed with {} articles'.format(len(failed)))
	
	wiki = WikiGraph_DB(args.db_path)

	todo = set(args.seeds)
	todo.update(wiki.inserted)
	todo = todo - wiki.expanded

	print('Initial todo queue: {}'.format(len(todo)))

	map_fn = map
	pool = None
	if args.num_workers > 0:
		pool = mp.Pool(args.num_workers)

		map_fn = pool.map

	bytes_downloaded = 0

	num = 0
	next_print = 0
	print_freq = 250
	while len(todo) > 0:

		start = time.time()

		B = args.batch_size if args.batch_size <= len(todo) else len(todo)
		
		article_names = list(itertools.islice(todo, B))

		info = map_fn(worker_expand, article_names)

		download_time = time.time() - start
		start = time.time()

		for out in info:

			if len(out) == 2: # failed

				print('F: {}'.format(out[0]))

				if out[0] not in failed:

					failed.add(out[0])
					failed_file.write(out[0] + '\n')
					failed_file.flush()

			else:

				article_name, links, redirs, bytes = out

				bytes_downloaded += bytes
				num += 1

				wiki.add(article_name, links, redirs)

				todo.update([link for link in links if link not in wiki.expanded])


		for name in article_names:
			todo.remove(name)

		save_time = time.time() - start

		if num >= next_print:
			print('\n** Num {}: bytes={}, todo={}, inserted={}, expanded={}, redirs={}, failed={}, recent={}, download={:.3f}, save={:.3f}'.format(
				num, bytes_downloaded, len(todo), len(wiki.inserted), len(wiki.expanded), len(wiki.inserted_redirs), len(failed),
				article_names[-1], download_time, save_time))
			next_print += print_freq
		
		# if itr == 10:
		# 	break
	
	print('\nAfter expanding {} articles - Final size: {}'.format(num, len(wiki)))
	
	del wiki
	failed_file.close()
	
	print('Done.')
	


if __name__ == '__main__':
	main()


	

