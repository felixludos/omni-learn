import os, sys

import argparse


MY_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
        # path = os.path.join(train.DEFAULT_DATA_PATH, 'emnist')
        # torchvision.datasets.EMNIST(path, split='letters', download=True, train=True)

        # mp.freeze_support()
        # mp.set_start_method('spawn')
        # a = torch.arange(3)#.cuda()
        # p = mp.Process(target=t, args=(a,1,))
        # p.start()
        # p.join()


        print(sys.argv)

        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('test', type=str)
        parser.add_argument('--test2', type=str)

        args = parser.parse_known_args()
        print(args)


        # train.register_config('mnist', os.path.join(MY_PATH, 'mnist', 'config', 'base.toml'))
        #
        # print(train.config._config_registry)
        # C = train.get_config('mnist')
        # print(C.keys())
