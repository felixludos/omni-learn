from .imports import *
from ..core import Mechanism as _Mechanism
from ..machines import Machine as _Machine, Event as _Event



class Machine(Configurable, _Machine):
	@fig.config_aliases(gap='app')
	def __init__(self, gap=None, **kwargs):
		super().__init__(gap=gap, **kwargs)



class Event(Machine, _Event):
	pass



class Mechanism(Machine, _Mechanism):
	def __init__(self, content: Union[AbstractGadget, Iterable[AbstractGadget]], *,
				 apply: Union[Dict[str, str], List[str]] = None,
				 select: Union[Dict[str, str], List[str]] = None,
				 insulate_out: bool = True, insulate_in: bool = True,
				 **kwargs):
		super().__init__(content=content, apply=apply, select=select,
				   insulate_in=insulate_in, insulate_out=insulate_out, **kwargs)




