import torch
import torch.multiprocessing as mp

def t(x, i=None):
        print( x+1 if i is None else x+i)


if __name__ == '__main__':
        mp.freeze_support()
        mp.set_start_method('spawn')
        a = torch.arange(3)#.cuda()
        p = mp.Process(target=t, args=(a,1,))
        p.start()
        p.join()    
