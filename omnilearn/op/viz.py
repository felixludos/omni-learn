from .imports import *
from omniply.visual import TraceRecorder as _TraceRecorder


class TraceRecorder(_TraceRecorder, fig.Configurable):
	pass

