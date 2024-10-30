from .imports import *
from omnilearn import *
from omnilearn import autoreg
from torchvision.datasets import MNIST as Torchvision_MNIST



fig.component('trainer')(Trainer)
fig.component('planner')(Planner)
fig.component('reporter')(Reporter)
fig.component('checkpointer')(Checkpointer)


@fig.component('mnist')
class MNIST(Dataset):
    _val_split = 10000
    def __init__(self, split: str = 'train', download: bool = True, **kwargs):
        super().__init__(**kwargs)
        if self._val_split is None:
            assert split in ('train', 'test'), f'Invalid split: {split}'
            size = 60000 if split == 'train' else 10000
        else:
            assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
            assert 0 < self._val_split < 60000, f'Invalid val_split: {self._val_split}'
            size = {'train': 60000-self._val_split, 'val': self._val_split, 'test': 10000}[split]

        self._split = split
        self._download = download
        self._size = size
        self._dataset = None
        self._image_data = None
        self._label_data = None


    def as_eval(self, **kwargs) -> 'MNIST':
        assert self._split == 'train', 'Only train split can be converted to eval'
        return self.__class__(split='val', **kwargs)


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
        return f'MNIST-{self._split}'
    

    @property
    def dataroot(self) -> str:
        return my_root / 'data'

    
    def load(self, *, device: str = None) -> Self:
        if self._dataset is None:
            self._dataset = Torchvision_MNIST(self.dataroot, train=self._split != 'test', download=self._download)
            self._image_data = self._dataset.data
            self._label_data = self._dataset.targets
            if self._split != 'test' and self._val_split is not None:
                if self._split == 'train':
                    self._image_data = self._image_data[self._val_split:]
                    self._label_data = self._label_data[self._val_split:]
                else:
                    self._image_data = self._image_data[:self._val_split]
                    self._label_data = self._label_data[:self._val_split]
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
    

    @tool('correct')
    def get_correct(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return (prediction.argmax(dim=1) == label)


    @tool('accuracy')
    def get_accuracy(self, correct: torch.Tensor) -> torch.Tensor:
        return correct.float().mean()




@fig.script('train')
def train_mnist(cfg: fig.Configuration):

    dataset: Dataset = cfg.pull('dataset')

    cfg.push('trainer._type', 'trainer', overwrite=False, silent=True)
    cfg.push('planner._type', 'planner', overwrite=False, silent=True)
    cfg.push('reporter._type', 'reporter', overwrite=False, silent=True)
    trainer: Trainer = cfg.pull('trainer')

    trainer.fit(dataset)
    
    return trainer



