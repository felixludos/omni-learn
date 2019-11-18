
import os
from ... import util
from .. import register_config

# print('loaded')
register_config('fdbase', os.path.join(util.FD_PATH, 'train', 'config_tml', 'base.toml'))

