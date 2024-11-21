from .imports import *
from ..core.viz import VizBatch as _VizBatch, VizContext as _VizContext, VizMechanism as _VizMechanism
from .common import Batch, Context, Mechanism

class VizBatch(_VizBatch, Batch):
    pass

class VizContext(_VizContext, Context):
    pass

class VizMechanism(_VizMechanism, Mechanism):
    pass
