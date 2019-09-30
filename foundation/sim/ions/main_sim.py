
import sys
import numpy as np
#import matplotlib.pyplot as plt
import config_parser as config
from forces import E_Field, Coulomb_Repulsion, Laser_Cooling
from simulation import *
from lab_setup import *
from stats import Energy, Temperature

args = None

def main(argv):

    parser = config.make_parser()
    global args
    args = parser.parse_args()

    



if __name__ == '__main__':
    sys.exit(main(sys.argv))