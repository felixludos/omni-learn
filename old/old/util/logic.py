
# from toposort import toposort as toposort_lib, toposort_flatten



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


