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
from extraction import get_redirs, get_links, wiki_collect, get_all_redirs

import configargparse


def init_expander(host, port, **unused):
	out = {
		'meta_db': redis.Redis(host=host, port=port, db=0),
		'articles_db': redis.Redis(host=host, port=port, db=1),
		'links_db': redis.Redis(host=host, port=port, db=2),
		'redirs_db': redis.Redis(host=host, port=port, db=3),
	}
	return out


def expand(meta_db, articles_db, links_db, redirs_db, **unused):

	try:
		names = []
		for _ in range(50):
			x = meta_db.spop('redir')
			if x is None:
				break
			names.append(x)
		
		mps = { m['from'].encode() : m['to'].encode() for m in get_all_redirs(*names)}
		
		for name in names:
			if name in mps:
				# an article
				redirs_db.set(name, mps[name])
				meta_db.sadd('redirs', name)
				meta_db.sadd('articles', mps[name])
			else:
				# a redirect
				meta_db.sadd('articles', name)

		return len(names) - len(mps) # num new articles

	except Exception as e:
		meta_db.sadd('failed', *names)
		#print('**failed: {}'.format(name))
		raise e

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
	
	private_args = {
		'host': args.host,
		'port': args.port,
	}

	full_remaining = meta_db.scard('redir')
	remaining = full_remaining
	num = 0
	start = time.time()
	smooth_time = None

	next_print = 0
	print_freq = 1000

	while remaining > 0:

		try:

			wiki = util.Farmer(fn=expand, private_args=private_args, num_workers=args.num_workers,
							   init_fn=init_expander, auto_dispatch=False, timeout=20)

			while remaining > 0:

				#start = time.time()

				B = args.batch_size if args.batch_size <= remaining else remaining

				wiki._dispatch(B)

				for _ in range(B):
					next(wiki)

				#batch_time = time.time() - start

				#smooth_time = batch_time if smooth_time is None else (smooth_time*0.99 + batch_time*0.01)

				remaining = meta_db.scard('redir')
				num = full_remaining - remaining

				if num >= next_print:
					print('\nNum {}/{}: remaining={}, articles={}, redirs={}, failed={}, time={:.3f}'.format(
						num, full_remaining, remaining, meta_db.scard('articles'), meta_db.scard('redirs'),
						meta_db.scard('failed'), (time.time() - start)/num))
					next_print += print_freq

		except Exception as e:
			print('**failed: {}'.format(e))
			#print('** failed')
			time.sleep(10)
			print('-- sleep --')
			
	print('Saving all databases...')
	meta_db.save()
	
	print('Done.')
	


if __name__ == '__main__':
	main()


	

