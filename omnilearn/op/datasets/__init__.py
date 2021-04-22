from .mnist import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, KMNIST, EMNIST
from .disentangling import dSprites, Shapes3D, CelebA, MPI3D
from .transforms import Concat, Cropped, Interpolated, Resamplable
from .mvtec import MVTec_Anomaly_Detection
from .translation import UnpairedTranslationDataset, BatchedUnpairedTranslationDataset
from .simple import FunctionDataset, FunctionSamples