from .imports import *
from omnilearn import *
from omnilearn.op import *
from omnilearn import autoreg
from omnilearn import scripts
from torchvision.datasets import MNIST as Torchvision_MNIST



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
    def size(self) -> int:
        return self._size
    

    @property
    def name(self) -> str:
        return 'MNIST' if self._split == 'train' else f'MNIST-{self._split}'
    

    @property
    def dataroot(self) -> str:
        return my_root / 'data'

    
    def setup(self, *, device: str = None):
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
        return super().setup(device=device)


    @tool('image')
    def get_images(self, indices: np.ndarray) -> torch.Tensor:
        '''returns int8 tensor of shape (N, 28, 28)'''
        return self._image_data[torch.from_numpy(indices)]
    @get_images.space
    def image_space(self) -> spaces.Pixels:
        return spaces.Pixels(1, 28, 28, as_bytes=True)
    

    @tool('label')
    def get_labels(self, indices: np.ndarray) -> torch.Tensor:
        return self._label_data[torch.from_numpy(indices)]
    @get_labels.space
    def label_space(self) -> spaces.Categorical:
        return spaces.Categorical(10)



@fig.component('classification')
class ImageClassification(Machine):
    @tool('observation')
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        N, *_ = image.shape
        if image.dtype == torch.uint8:
            image = image.float().div(255)
        return image.view(N, -1)
    @transform_image.space
    def observation_space(self, image: spaces.Pixels) -> spaces.Bounded:
        """Flattens the image"""
        return spaces.Bounded(image.size, lower=0., upper=1.)


    @space('prediction')
    def prediction_space(self, label: spaces.Categorical) -> spaces.Logits:
        return spaces.Logits(label.n)


    label_space = space('label')
    

    @indicator('loss')
    def get_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(prediction, label)
    

    @tool('correct', space=spaces.Boolean(1))
    def get_correct(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return (prediction.argmax(dim=1) == label)


    @indicator('accuracy')
    def get_accuracy(self, correct: torch.Tensor) -> torch.Tensor:
        return correct.float().mean()


    def settings(self) -> Dict[str, Any]:
        return {'observation': self.observation_space.json(), 'label': self.label_space.json()}



# @fig.script('train')
def train_mnist(cfg: fig.Configuration):

    record_step = cfg.pull('record-step', False)
    if record_step:
        Dataset._Batch = VizBatch
        # Trainer._Batch = VizBatch
        fig.component('mechanism')(VizMechanism)

    dataset: Dataset = cfg.pull('dataset')

    cfg.push('trainer._type', 'trainer', overwrite=False, silent=True)
    cfg.push('planner._type', 'planner', overwrite=False, silent=True)
    cfg.push('reporter._type', 'reporter', overwrite=False, silent=True)
    trainer: Trainer = cfg.pull('trainer')

    self = trainer

    reporter = self.reporter

    system = self.prepare(src)

    plan = self.plan(system)
    reporter.begin(self, plan)

    batch_cls = self._Batch or getattr(src, '_Batch', None) or Batch
    for info in plan.generate(batch_size):
        batch = batch_cls(info, plan=plan)
        if system is not None:
            batch.include(system)

        terminate = self.learn(batch)
        if terminate: break

    reporter.end(batch)




    if record_step:
        for batch in trainer.fit_loop(dataset): break
        print()
        print(batch.report(**cfg.pull('report-settings', {})))
    else:
        trainer.fit(dataset)
    
    return trainer






