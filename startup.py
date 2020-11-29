
# import foundation

import omnifig as fig


@fig.Script('test-script')
def test_script(A):
	
	import foundation as fd
	from foundation import util
	
	import matplotlib.pyplot as plt
	
	import torch
	from torch import nn
	
	print('running test script in foundation')
	
	A = fig.get_config()
	
	A.push('clock._type', 'clock')
	A.push('clock._mod.clock-stats', 1)
	
	A.push('scheduler._type', 'scheduler/step')
	A.push('scheduler.freq', 10)
	A.push('scheduler.gamma', 0.5)
	# A.push('scheduler._mod.alert/reg', 1)
	
	clock = A.pull('clock')
	
	sch = A.pull('scheduler')
	
	clock.register_alert('scheduler', sch)
	
	model = nn.Linear(10, 4)
	optim = util.get_optimizer('adam', model.parameters(), lr=1)
	
	sch.include_optim(optim)
	
	lrs = []
	
	for _ in range(100):
		clock.tick()
		lrs.append(optim.param_groups[0]['lr'])
		
	plt.plot(lrs)
	
	pass