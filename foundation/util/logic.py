
# from toposort import toposort as toposort_lib, toposort_flatten
from c3linearize import linearize
from collections import deque

class CycleDetectedError(Exception):
	def __init__(self, node):
		super().__init__('Cycle detected near: {}'.format(node))

# graph = {0:[1,2], 1:[3,4], 2:[4,6], 3:[7], 4:[5], 5:[], 6:[], 7:[]}
# util.toposort(0,lambda x: graph[x], depth_first=True)
# produces: [0, 1, 3, 7, 2, 4, 6, 5]
# but should be: [0, 1, 3, 7, 2, 4, 5, 6]

def graph_conv(x, g, d=None):
	if d is None:
		d = {}
	if x not in d:
		e = g(x)
		if x not in d:
			d[x] = e
		for v in e:
			graph_conv(v, g, d)
	return d
	
		
def toposort(root, get_edges, ordered=True):
	src = graph_conv(root, get_edges)
	
	return linearize(src, heads=[root], order=ordered)[root]
	


def _toposort_bad(root, get_edges, depth_first=False):

	if depth_first:
		raise NotImplementedError # not working atm

	order = [root]
	done = set(order)
	options = deque(get_edges(root))

	while len(options):

		next = None
		for node in options:
			success = True
			for check in options:
				if node in get_edges(check):
					success = False
					break
			if success:
				next = node
				break
		if next is None:
			raise CycleDetectedError(node)
		else:
			options.remove(next)
			if next not in done:
				order.append(next)
				done.add(next)
				if depth_first:
					options.extendleft(reversed(get_edges(next)))
				else:
					options.extend(get_edges(next))

	return order



# def toposort(root, get_edges, depth_first=False):
#
# 	order = [root]
# 	done = set(order)
# 	options = deque(get_edges(root))
#
# 	while len(options):
#
# 		next = None
# 		for node in options:
# 			success = True
# 			for check in options:
# 				if node in get_edges(check):
# 					success = False
# 					break
# 			if success:
# 				next = node
# 				break
# 		if next is None:
# 			raise CycleDetectedError(node)
# 		else:
# 			options.remove(next)
# 			if next not in done:
# 				order.append(next)
# 				done.add(next)
# 				if depth_first:
# 					options.extendleft(reversed(get_edges(next)))
# 				else:
# 					options.extend(get_edges(next))
#
# 	return order


