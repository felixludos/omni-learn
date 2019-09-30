

import sys, os, time
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import gym
import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import data



class DirectDecoder(fd.Generative_Model, fd.Unsupervised_Model):


    def __init__(self, latent_dim, out_shape, vocab_size, **dec_kwargs):
        super().__init__(nn.BCELoss(), latent_dim, out_shape)

        self.latent_dim = latent_dim
        self.out_shape = out_shape
        self.out_dim = np.product(out_shape)

        self.table = nn.Embedding(vocab_size, latent_dim)

        self.dec = models.Decoder(out_shape, latent_dim=latent_dim, **dec_kwargs)



    def forward(self, idx):

        q = self.table(idx)
        out = self.decode(q)
        return out

    def get_loss(self, sample, stats=None): # reconstruction only

        idx, (x, _) = sample

        pred = self(idx)

        out = {
            'loss': self.criterion(pred, x),
            'original': x,
            'reconstruction': pred.detach(),
        }

        return out

    def decode(self, q):
        return self.dec(q)

    def generate(self, N=1):
        raise NotImplementedError






def main():
    args = util.NS()

    args.device = 'cuda:0'
    args.seed = 0

    args.logdate = True
    args.tblog = True
    args.txtlog = False
    args.saveroot = '../trained_nets'

    args.dataset = 'mnist'
    args.indexed = True
    args.use_val = False

    args.num_workers = 0
    args.batch_size = 128

    args.start_epoch = 0
    args.epochs = 10

    args.name = 'test_on_mnist'
    args.latent_dim = 2

    args.save_model = False

    now = time.strftime("%y-%m-%d-%H%M%S")
    if args.logdate:
        args.name = os.path.join(args.name, now)
    args.save_dir = os.path.join(args.saveroot, args.name)
    print('Save dir: {}'.format(args.save_dir))

    if args.tblog or args.txtlog:
        util.create_dir(args.save_dir)
        print('Logging in {}'.format(args.save_dir))
    logger = util.Logger(args.save_dir, tensorboard=args.tblog, txt=args.txtlog)

    # Set seed
    if not hasattr(args, 'seed') or args.seed is None:
        args.seed = util.get_random_seed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        torch.cuda.manual_seed(args.seed)
    except:
        pass

    if not torch.cuda.is_available():
        args.device = 'cpu'
    print('Using device {} - random seed set to {}'.format(args.device, args.seed))

    datasets = train.load_data(args=args)

    shuffles = [True, False, False]

    loaders = [DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=s) for d, s in
               zip(datasets, shuffles)]

    trainloader, testloader = loaders[0], loaders[-1]
    valloader = None if len(loaders) == 2 else loaders[1]

    print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
    if valloader is not None:
        print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
    print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
    print('Batch size: {} samples'.format(args.batch_size))

    # Define Model
    args.total_samples = {'train': 0, 'test': 0}
    epoch = 0
    best_loss = None
    all_train_stats = []
    all_test_stats = []

    model = DirectDecoder(latent_dim=args.latent_dim, out_shape=(1, 28, 28), vocab_size=len(datasets[0]),

                          nonlin='prelu', output_nonlin='sigmoid',
                          channels=[8, 16, 32], kernels=[3, 3, 3], ups=[2, 2, 2], upsampling='bilinear',
                          output_norm_type=None,
                          hidden_fc=[32],
                          )

    model.to(args.device)
    print(model)
    print('Model has {} parameters'.format(util.count_parameters(model)))

    optim = util.get_optimizer('adam', model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    scheduler = None  # torch.optim.lr_scheduler.StepLR(optim, step_size=6, gamma=0.2)

    for sample in iter(trainloader):
        idx, (x,y) = sample
        idx, (x,y) = idx.to(args.device), (x.to(args.device), y.to(args.device))
        break

    out = model(idx)

    print(out.shape)

    loss = model.get_loss(x, idx)

    print(loss)

    print('test complete.')

if __name__ == '__main__':

    main()
