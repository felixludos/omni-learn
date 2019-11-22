
import configargparse
from lab_setup import Barium, Ytterbium 

def make_parser():
    p = configargparse.ArgParser(description='Simulate trapped Ba or Yb Ions in a linear Paul Trap')
    p.add('-c', '--config_tml', required=True, is_config_file=True, help='config_tml file path')

    p.add('--name', type=str, required=True, help='Name of simulation')
    p.add('--save-root', type=str, required=True, help='Root for saving results')
    p.add('--load', type=str, default=None, help='Root for loading a sim state to continue that simulation') # will disregard most settings below
    p.add('--save-xyz', action='store_true', help='Save .xyz file of positions to view trajectory')
    p.add('--save-spectra', action='store_true', help='Save freq spectra in X,Y,Z dirs')
    p.add('--save-stats', action='store_true', help='Save collected stats (temp, energy)')
    p.add('--frames', type=int, required=True, help='Total number of timesteps to simulate')
    p.add('--print-step', type=int, default=500, help='Number of timesteps to simulate between saving data (default: 500)')
    p.add('--seed', type=int, default=None, help='Seed for random number generator')
    p.add('--init-setting', type=str, choices=['eq', 'random'], help='Setting for initial pos and vel (options: eq, random)')
    p.add('--no-cooling', dest='cooling', action='store_false', help='Don\'t use cooling lasers')
    p.add('--timestep', type=str, default='auto', help='Timestep in sec (should be ~ 1 ns), set to \'auto\' to automatically compute a good timestep based on trap freq (~40 frames per trap cycle)')
    p.add('--trap-V_0', type=float, default=1000, help='Radial (xy) E-field amplitude in Volts (default 1000)')
    p.add('--trap-V_caps', type=float, default=10, help='Axial (z) E-field amplitude in Volts (default 10')
    p.add('--trap-w_E', type=float, default=18.2e6, help='Radial E-field frequency in Hz (default 18.2e6)')
    p.add('--trap-X', type=float, default=0.93e-3, help='X trap length in m (default 0.93e-3)')
    p.add('--trap-Y', type=float, default=0.793e-3, help='Y trap length in m (default 0.793e-3)')
    p.add('--trap-Z', type=float, default=2.84e-3, help='Z trap length in m (default 2.84e-3)')
    p.add('--trap-cooling-lasers', type=str, choices=['all-dirs', 'single'], help='Setup for Ba cooling lasers (options: \'all-dirs\', \'single\')')
    p.add('--ions', type=str, default='Ba', help='Trapped ion element symbols in order from +Z to -Z as CSV (default: \'Ba\' ')

    return p

def parse_ion_list(ls_str):
    ls = ls_str.split(',')
    ions = []
    for sym in ls:
        if sym == 'Ba':
            ions.append(Barium)
        elif sym == 'Yb':
            ions.append(Ytterbium)
        else:
            raise Exception('Ion list could not be parsed:' + str(ls_str))
    return ions



