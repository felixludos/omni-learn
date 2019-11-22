
import os
from ... import util
from .. import register_config

# print('loaded')
register_config('base', os.path.join(util.FD_PATH, 'train', 'base_config', 'base.yaml'))

