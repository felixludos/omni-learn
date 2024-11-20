from .imports import *



def parameter_count(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)




