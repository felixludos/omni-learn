
import sys
from foundation.old import train as trn

if __name__ == '__main__':
	
	A = trn.parse_config()
	
	path = A.pull('path', None)
	keep_ckpts = A.pull('keep_ckpts', 'last')
	purge_tests = A.pull('purge_tests', False)
	
	sys.exit(trn.sanitize_runs(path=path, keep_ckpts=keep_ckpts, purge_tests=purge_tests))


