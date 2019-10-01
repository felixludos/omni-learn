

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


class MLP_Decoder(fd.Decodable, fd.Model):

    def __init__(self, latent_dim, out_shape, hidden_dims=[], nonlin='prelu', output_nonlin=None):
        super().__init__(latent_dim, out_shape)

        self.out_shape = out_shape
        self.out_dim = np.product(out_shape)

        self.net = models.make_MLP(latent_dim, self.out_dim, hidden_dims=hidden_dims,
                                   output_nonlin=output_nonlin, nonlin=nonlin)

    def forward(self, q):
        return self.decode(q)

    def decode(self, q):
        B = q.size(0)
        out = self.net(q)
        return out.view(B, *self.out_shape)

class DirectDecoder(fd.Generative, fd.Decodable, fd.Vizualizable, fd.Trainable_Model):

    def __init__(self, decoder, latent_dim, vocab_size,
                 normalize_latent=False, zero_embedding=False,):
        super().__init__(decoder.din, decoder.dout)

        self.latent_dim = latent_dim

        self.table = nn.Embedding(vocab_size, latent_dim)
        if zero_embedding:
            self.table.weight.data.mul_(0)

        self.dec = decoder
        # self.dec = models.Decoder(out_shape, latent_dim=latent_dim, **dec_kwargs)

        self.normalize_latent = normalize_latent

        self.criterion = nn.BCELoss()

    def _visualize(self, info, logger):

        if self._viz_counter % 5 == 0:

            logger.add('histogram', 'latent-norm', info.latent.norm(p=2, dim=-1))

            B, C, H, W = info['original'].shape
            N = min(B, 8)

            viz_x, viz_rec = info['original'][:N], info['reconstruction'][:N]

            recs = torch.cat([viz_x, viz_rec],0)
            logger.add('images', 'rec', recs)

            # show some generation examples
            if self.normalize_latent:
                gen = self.generate(16)
                logger.add('images', 'gen', gen)


    def forward(self, q):
        return self.decode(q)

    def _step(self, batch):

        idx, (x, _) = batch

        q = self.retrieve(idx)
        pred = self.decode(q)

        loss = self.criterion(pred, x)

        self.optim_step(loss)

        out = util.TensorDict({
            'loss': loss.detach(),
            'original': x,
            'reconstruction': pred.detach(),
            'latent': q.detach(),
        })

        return out

    def retrieve(self, idx):
        return self.table(idx)

    def decode(self, q):
        # possibly normalize here
        if self.normalize_latent:
            q = F.normalize(q, p=2, dim=1)

        return self.dec.decode(q)

    def generate(self, N=1):

        assert self.normalize_latent, 'Must normalize to sample'

        q = torch.randn(N, self.latent_dim).to(self.table.weight.device)
        return self.decode(q)





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

    args.num_workers = 0#4
    args.batch_size = 128

    args.start_epoch = 0
    args.epochs = 10

    args.name = 'direct_decoder'
    args.latent_dim = 3

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

    loaders = train.get_loader(*datasets, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, drop_last=False, )

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
                          channels=[32, 16, 8], kernels=[3, 3, 3], ups=[2, 2, 2], upsampling='bilinear',
                          output_norm_type=None,
                          hidden_fc=[128],
                          )

    model.to(args.device)
    print(model)
    print('Model has {} parameters'.format(util.count_parameters(model)))

    model.set_optim(optim_type='adam', lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # optim = util.get_optimizer('sgd', model.parameters(), )
    scheduler = None  # torch.optim.lr_scheduler.StepLR(optim, step_size=6, gamma=0.2)

    lr = model.optim.param_groups[0]['lr']
    for _ in range(10):

        old_lr = lr
        if scheduler is not None:
            scheduler.step()
        lr = model.optim.param_groups[0]['lr']

        if lr != old_lr:
            print('--- lr update: {:.3E} -> {:.3E} ---'.format(old_lr, lr))

        train_stats = util.StatsMeter('lr', tau=0.1)
        train_stats.update('lr', lr)

        train_stats = train.run_epoch(model, trainloader, args, mode='train',
                                      epoch=epoch, print_freq=10, logger=logger, silent=True,
                                      viz_criterion_args=['reconstruction', 'original'],
                                      stats=train_stats)

        all_train_stats.append(train_stats)

        print('[ {} ] Epoch {} Train={:.3f} ({:.3f})'.format(
            time.strftime("%H:%M:%S"), epoch + 1,
            train_stats['loss-viz'].avg.item(), train_stats['loss'].avg.item(),
        ))

        if args.save_model:

            av_loss = train_stats['loss'].avg.item()
            is_best = best_loss is None or av_loss < best_loss
            if is_best:
                best_loss = av_loss

            path = train.save_checkpoint({
                'epoch': epoch,
                'args': args,
                'model_str': str(model),
                'model_state': model.state_dict(),
                'all_train_stats': all_train_stats,
                'all_test_stats': all_test_stats,

            }, args.save_dir, is_best=is_best, epoch=epoch + 1)
            print('--- checkpoint saved to {} ---'.format(path))

        epoch += 1

    print('Done')

    print('test complete.')

if __name__ == '__main__':

    main()
