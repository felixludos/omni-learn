
import os

from omnifig import register_config

from foundation import util

# print('loaded')
register_config('origin', os.path.join(util.FD_PATH, 'op', 'base_config', 'origin.yaml'))

