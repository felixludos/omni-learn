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
import queue

import redis
from extraction import wiki_expand

import configargparse


def init_expander(host, port, **unused):
	out = {
		'meta_db': redis.Redis(host=host, port=port, db=0),
		'articles_db': redis.Redis(host=host, port=port, db=1),
		'links_db': redis.Redis(host=host, port=port, db=2),
		'redirs_db': redis.Redis(host=host, port=port, db=3),
	}
	return out


def expand(meta_db, articles_db, links_db, redirs_db, name, **unused):

	tries = 0

	while True:
		try:

			links, redirs, _, bts = wiki_expand(name)
			groups = [[link['href'][6:].split('#')[0].encode() for link in group] for group in [links, redirs]]
			links, redirs = map(set, groups)
			if name in links:
				links.remove(name)

		except URLError as e:

			if e.code == 429:

				# print(e.read())
				# print(e.headers)
				# raise e

				#print('**Too many requests: sleeping for 5 sec')
				time.sleep(5)
			else:
				meta_db.sadd('failed', name)

				return name, e.code
		else:
			break
		tries += 1

		if tries > 10:
			raise Exception('Tried too many times')
	
	# articles
	
	meta_db.sadd('articles', name)
	
	if len(links):
		meta_db.sadd('articles', *links)
		links_db.sadd(name, *links)
	
	if len(redirs):
		meta_db.sadd('redirs', *redirs)
		redirs_db.sadd(name, *redirs)
	
	for link in links:
		articles_db.incr(link)
	
	return name, links, len(redirs), bts

args = None

def main():
	# len(links), len(redirs)
	parser = configargparse.ArgumentParser(description='Wikipedia Collect Article Links using redis')
	
	# parser.add_argument('--seeds', type=str, default=['Artificial_intelligence', 'Computational_physics', 'Control_theory', 'Machine_learning',
	#          'Quantum_computing'], nargs='+')
	
	parser.add_argument('--seeds', type=str,
						default=[b'Artificial_intelligence', ], nargs='+')
	#parser.add_argument('--db-path', default='test.zeo', type=str)
	parser.add_argument('--host', type=str, default='localhost')
	parser.add_argument('--port', type=int, default=6379)
	parser.add_argument('-j','--num-workers', default=4, type=int)
	parser.add_argument('--batch-size', default=16, type=int)	

	
	global args
	args = parser.parse_args()

	args.host = args.host.encode() # convert to bytes
	
	print('host={}, port={}'.format(args.host, args.port))
	
	meta_db = redis.Redis(host=args.host, port=args.port, db=0)
	articles_db = redis.Redis(host=args.host, port=args.port, db=1)
	links_db = redis.Redis(host=args.host, port=args.port, db=2)
	redirs_db = redis.Redis(host=args.host, port=args.port, db=3)

	todo = set(args.seeds)
	
	inserted = set(meta_db.smembers('articles'))
	expanded = set(links_db.keys())
	failed = set(meta_db.smembers('failed'))
	
	todo.update(inserted)
	todo = todo - expanded
	todo = todo - failed
	print('Initial todo queue: {}'.format(len(todo)))
	print('Initial inserted: {}'.format(len(inserted)))
	print('Initial expanded: {}'.format(len(expanded)))
	
	bytes_downloaded = 0
	total_redirs = redirs_db.dbsize()
	
	private_args = {
		'host': args.host,
		'port': args.port,
	}
	
	wiki = util.Farmer(fn=expand, private_args=private_args, num_workers=args.num_workers,
	                   init_fn=init_expander, auto_dispatch=False, timeout=40)

	
	smooth_time = None

	link_lens = 0

	num = 0
	next_print = 0
	next_save = 0
	print_freq = 500
	while len(todo) > 0:

		start = time.time()

		B = args.batch_size if args.batch_size <= len(todo) else len(todo)
		
		article_names = list(itertools.islice(todo, B))
		
		wiki.volatile_gen = iter([{'name':name} for name in article_names])
		wiki._dispatch(B)

		f_count = 0
		N = 0

		try:

			for out in wiki:

				if len(out) == 2: # failed

					f_count += 1
					if f_count > 10:
						raise Exception('failed too many times')

					print('** Error {}: {}'.format(out[1], out[0]))
					failed.add(out[0])

				else:

					article_name, links, num_redirs, bytes = out

					if len(links) == 0:  # failed

						print('** Error {}: {}'.format('LINK', article_name))
						meta_db.sadd('failed', article_name)
						failed.add(out[0])

					else:

						bytes_downloaded += bytes
						num += 1
						total_redirs += num_redirs
						link_lens += len(links)

						expanded.add(article_name)
						inserted.update(links)
						todo.update([link for link in links if link not in expanded])

				N += 1

		except queue.Empty as e:

			print(wiki.outstanding)
			print(N)

			raise e


		for name in article_names:
			todo.remove(name)

		batch_time = time.time() - start
		
		smooth_time = batch_time if smooth_time is None else (smooth_time*0.99 + batch_time*0.01)

		if num >= next_print:
			print('\nNum {}: MB={:.3f}, todo={}, inserted={}, expanded={}, redirs={}, links={}, failed={}, recent={}, time={:.3f} ({:.3f})'.format(
				num, bytes_downloaded/1000000, len(todo), len(inserted), len(expanded), total_redirs, link_lens, len(failed),
				article_names[-1], batch_time/len(article_names), smooth_time/len(article_names)))
			next_print += print_freq
	
	print('\nAfter expanding {} articles - Final size: {}'.format(num, len(links_db.dbsize())))
	
	
	print('Saving all databases...')
	meta_db.save()
	articles_db.save()
	links_db.save()
	redirs.db.save()
	
	print('Done.')
	


if __name__ == '__main__':
	main()


	

