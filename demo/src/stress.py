from .imports import *
from .op import *




@fig.component('meaner')
class Meaner(Machine):
    @tool('selection')
    def get_selection(self, observation: torch.Tensor) -> torch.Tensor:
        B = observation.shape[0]
        return observation.view(B, -1).mean(dim=1).view(B, 1)
    @get_selection.space
    def selection_space(self, observation: spaces.Bounded) -> spaces.Bounded:
        return spaces.Vector(1)
    


@fig.component('bernies')
class Bernies(MNIST):
    def __init__(self, split: str = 'train', mean_only: bool = True, download: bool = True, **kwargs):
        super().__init__(split=split, download=download, **kwargs)
        self._mean_only = mean_only
    
    
    def load(self, *, device: str = None) -> Self:
        super().load(device=device)
        if self._split == 'train':
            lbls = self._label_data
            imgs = self._image_data.float().div(255)
            means = []
            for i in range(10):
                idx = lbls == i
                sel = imgs[idx]
                mu = sel.mean(0, keepdim=True)
                means.append(mu)
                imgs[idx] = torch.rand_like(sel).lt(mu if self._mean_only else sel).float()
            self._means = torch.stack(means)
            self._image_data = imgs.mul(255).byte()




    pass




