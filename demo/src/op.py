from .imports import *
from omnilearn import Dataset, Model, Adam, Optimizer, Trainer, Machine, Planner
from omnilearn import autoreg
from torchvision.datasets import MNIST as Torchvision_MNIST



fig.component('trainer')(Trainer)
fig.component('planner')(Planner)


@fig.component('mnist')
class MNIST(Dataset):
    def __init__(self, train: bool = True, download: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._train = train
        self._download = download
        self._size = 60000 if train else 10000
        self._dataset = None
        self._image_data = None
        self._label_data = None


    @property
    def input_dim(self) -> int:
        return (28, 28)
    

    @property
    def output_dim(self) -> int:
        return 10


    @property
    def size(self) -> int:
        return self._size
    

    @property
    def name(self) -> str:
        return f'MNIST-{"train" if self._train else "test"}'
    

    @property
    def dataroot(self) -> str:
        return my_root / 'data'

    
    def load(self, *, device: str = None) -> Self:
        if self._dataset is None:
            self._dataset = Torchvision_MNIST(self.dataroot, train=self._train, download=self._download)
            self._image_data = self._dataset.data
            self._label_data = self._dataset.targets
            if device is not None:
                self._image_data = self._image_data.to(device)
                self._label_data = self._label_data.to(device)
        return super().load(device=device)


    @tool('image')
    def get_images(self, index: np.ndarray) -> torch.Tensor:
        '''returns int8 tensor of shape (N, 28, 28)'''
        return self._image_data[torch.from_numpy(index)]
    

    @tool('label')
    def get_labels(self, index: np.ndarray) -> torch.Tensor:
        return self._label_data[torch.from_numpy(index)]



@fig.component('classification')
class ImageClassification(Machine):
    @tool('observation')
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        N, *_ = image.shape
        if image.dtype == torch.uint8:
            image = image.float().div(255)
        return image.view(N, -1)
    

    @tool('loss')
    def get_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(prediction, label)



@fig.script('train')
def train_mnist(cfg: fig.Configuration):

    dataset: Dataset = cfg.pull('dataset')

    cfg.push('trainer._type', 'trainer', overwrite=False, silent=True)
    cfg.push('planner._type', 'planner', overwrite=False, silent=True)
    trainer: Trainer = cfg.pull('trainer')

    trainer.fit(dataset)
    
    return trainer



