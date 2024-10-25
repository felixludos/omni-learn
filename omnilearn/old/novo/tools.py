import omnifig as fig
from omnifig.config import ConfigManager


@fig.component('attr')
def get_attribute(cfg):
	key = cfg.pull('key')
	attr = cfg.pull('attr')

	obj = cfg.pull(key)
	return getattr(obj, attr)













