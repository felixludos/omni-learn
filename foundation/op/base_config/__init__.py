
import os

from omnifig import register_config

from foundation import util

# print('loaded')
register_config('base', os.path.join(util.FD_PATH, 'op', 'base_config', 'base.yaml'))

