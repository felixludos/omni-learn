from typing import Sequence, Union, Any, Optional, Tuple, Dict, List, Iterable, Iterator, Callable, Type, Set, Self

# from __future__ import annotations
from pathlib import Path
import sys, os, shutil
from datetime import datetime

from omnibelt import unspecified_argument, pformat, where_am_i
from omniply import AbstractGadget, AbstractGaggle, AbstractGame
# from omniply import Scope, Selection#, ToolKit, Context, tool
# from omniply.apps.gaps import ToolKit, Context, tool

import numpy as np

from tqdm import tqdm
from tabulate import tabulate
from humanize import naturalsize
import time, random, hashlib, heapq, math
from collections import Counter


# pytorch (used required by `compute` to init, but may be used inside others e.g. spaces.simple)
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim as O


