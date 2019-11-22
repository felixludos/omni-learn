
from collections import deque

class CycleDetectedError(Exception):
	def __init__(self, node):
		super().__init__('Cycle detected near: {}'.format(node))

def toposort(root, get_edges):

	order = [root]
	options = get_edges(root)

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
			order.append(next)
			options += get_edges(next)

	return order



