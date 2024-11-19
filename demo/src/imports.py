from typing import Union, List, Dict, Any, Tuple, Optional, Type, Iterable, Iterator, Self
import sys, os, shutil
from pathlib import Path
import subprocess
import omnifig as fig
from omnibelt import pformat, where_am_i
import random
from collections import Counter
import json
import h5py as hf
from tabulate import tabulate
from tqdm import tqdm 
from omniply import Scope, Selection
from omniply import AbstractGadget
# from omniply.apps.gaps import tool, ToolKit, Context#, Scope, Selection
from omniply.apps.training import Dataset as DatabaseBase, Batch, DynamicTrainerBase
from omnilearn import *

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from torch import nn
from torch.nn import functional as F

my_root = Path(__file__).parent.parent

