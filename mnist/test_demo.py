import os
import omnifig as fig
# import omnilearn as learn

def test_demo():
	root = os.path.dirname(__file__)
	print(root)
	fig.initialize(root)
	
	run = fig.quick_run('load-run', 'demo')
	
	print(run)
	print(run.__class__)
	print(run.__class__.__mro__)
	try:
		fig.quick_run(None, 'demo', budget=10)
	except AttributeError:
		print(fig.find_component('run'))
		print(run)
		print(run.__class__)
		print(run.__class__.__mro__)
		raise

