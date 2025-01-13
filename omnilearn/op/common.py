from .imports import *
from ..core import Mechanism as _Mechanism
from ..machines import Machine as _Machine, Event as _Event



class Machine(Configurable, _Machine):
	@fig.config_aliases(gap='app')
	def __init__(self, *args, gap=None, **kwargs):
		super().__init__(*args, gap=gap, **kwargs)


	def _prepare(self, *, device: str = None, **kwargs):
		out = super()._prepare(device=device, **kwargs)
		for gadget in self.vendors():
			if isinstance(gadget, Machine):
				gadget.prepare(device=device)
		return out


class Event(Machine, _Event):
	pass



class Mechanism(Configurable, Prepared, _Mechanism, AbstractMachine):
	def __init__(self, content: Union[AbstractGadget, Iterable[AbstractGadget]], *,
				 internal: Union[Dict[str, str], List[str]] = None,
				 external: Union[Dict[str, str], List[str]] = None,
				 exclusive: bool = True, insulated: bool = True,
				 **kwargs):
		super().__init__(*content, internal=internal, external=external,
				   insulated=insulated, exclusive=exclusive, **kwargs)

	def checkpoint(self, path = None):
		pass

	def settings(self):
		return {'content': [g.settings() for g in self.vendors()],
		  'internal': self._internal_map, 'external': self._external_map,
		  'exclusive': self._exclusive, 'insulated': self._insulated}

	def _prepare(self, *, device: str = None):
		out = super()._prepare(device=device)
		for gadget in self.vendors():
			if isinstance(gadget, Machine):
				gadget.prepare(device=device)
		return out




