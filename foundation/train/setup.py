

import sys, os, time
from .. import util


def setup_logging(info):

	if 'invisible' in info and info.invisible:
		print('No record of this run will be made')
		return None

	assert 'name' in info, 'This run is missing a name'

	if 'save_dir' not in info:
		now = time.strftime("%y%m%d-%H%M%S")
		if 'logdate' in info and info.logdate and '_logged_date' not in info:
			info.name = '{}_{}'.format(info.name, now)
			info._logged_date = True

		info.save_dir = os.path.join(os.environ['FOUNDATION_SAVE_DIR'], info.name)
	if 'save_dir' in info:
		print('Save dir: {}'.format(info.save_dir))
	else:
		info.tblog, info.txtlog = False, False
		print('WARNING: No save_dir, so there will be no record of this run')
	if info.tblog or info.txtlog:
		util.create_dir(info.save_dir)
		logtypes = []
		if info.txtlog:
			logtypes.append('stdout')
		if info.tblog:
			logtypes.append('on tensorboard')
		print('Logging {}'.format(' and '.join(logtypes)))

	logger = util.Logger(info.save_dir, tensorboard=info.tblog, txt=info.txtlog)

	return logger

def setup_records(training):

	records = {
		'total_samples': {'train':0, 'val':0, 'test':0, },
		'stats': {'train':[], 'val':[]},
	}

	if 'start' not in training:
		training.start = 0
	records['epoch'] = training.start

	if training.track_best:
		records['best'] = {'loss':None, 'epoch':None}

	if 'stats_decay' in training:
		util.set_default_tau(training.stats_decay)

	return records
