from .imports import *
from .containers import Context, Mechanism, Batch



class VizContext(Context, omniply_viz_Context):
	pass



class VizMechanism(Mechanism, omniply_viz_Mechanism):
	pass



class VizBatch(Batch, VizContext):
	def __init__(self, *args, start_recording: bool = True, **kwargs):
		'''can automatically start recording'''
		super().__init__(*args, **kwargs)
		if start_recording:
			self.record()

