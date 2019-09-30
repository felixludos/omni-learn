import numpy as np
from collections import namedtuple

# constants
c = 299792458 # m/s
h = 6.626070040e-34 # Js
h_bar = h / (2 * np.pi)
kB = 1.38064852e-23 # J/K
e = 1.6021766208e-19 # C
vacuum_permittivity = 8.854187817e-12 # F/m
K = 1. / (4 * np.pi * vacuum_permittivity) # N*(m/C)^2 - coulombs constant


# conversion factors
amu2kg =  1.660539040e-27

zero_vector = np.array([0.,0.,0.])
zero_vector.flags.writeable = False
def gen_unit_vector(sum_of=1):
    vs = np.random.rand(sum_of,3)*2 - 1
    norms = np.linalg.norm(vs, axis=1, keepdims=True)
    return np.sum(vs/norms,axis=0)
#def gen_unit_vector():
#    v = np.random.rand(3)*2 - 1
#    return v / np.linalg.norm(v)

# atoms
atomic_properties = ['name', # str
                     'symbol', # str
                     'number', # atomic number
                     'mass', # kg - atomic mass
                     'charge', # C - ion charge
                     'cooling', # bool - is there doppler cooling
                     'natural_line_width', # Hz - of cooling transition
                     'cooling_laser_wavelength', # Hz
                     'rabi_frequency'] # Hz
Atom = namedtuple('Atom', atomic_properties)

Barium = Atom(name='Barium',
              symbol='Ba',
              number=56,
              mass=137.327 * amu2kg,
              charge=e,
              cooling = True,
              natural_line_width = 2 * np.pi * 15e6,
              cooling_laser_wavelength = 493.4077e-9,
              rabi_frequency = 62.1012557471e6,
             )

Ytterbium = Atom(name='Ytterbium',
                 symbol='Yb',
                 number=70,
                 mass=173.04 * amu2kg,
                 charge=e,
                 cooling = False,
                 natural_line_width = 2 * np.pi * 40e6, # MISSING
                 cooling_laser_wavelength = -1, #369e-9,
                 rabi_frequency = -1, # MISSING
                 )

# laser setups
six_dirs = np.vstack([np.identity(3), -np.identity(3)])
single_laser = np.array([[1.,1.,1.]]) / np.sqrt(3)

# trap setups
trap_properties = ['w_E', # Hz - radial field frequency
                   'V_0', # V - radial field amplitude
                   #'V_dc', # V - radial field offset
                   'V_caps', # V - axial field dc
                   'X', # m - radius in X direction
                   'Y', # m - radius in Y direction
                   'Z', # m - radius in Z direction
                   'laser_setup', # cooling laser directions
                  ]
Ion_Trap = namedtuple('Ion_Trap', trap_properties)

Georgia_Tech_2007 = Ion_Trap(w_E = 2 * np.pi * 6.6e6,
                             V_0 = 50 * np.sqrt(2),
                             V_caps = 7.,
                             X = 1.85e-3 / 2,
                             Y = 1.85e-3 / 2,
                             Z = 10e-3 / 2,
                             laser_setup = six_dirs,
                            )

My_Trap = Ion_Trap(w_E = 2 * np.pi * 18.2e6,
                   V_0 = 10000,
                   V_caps = 1,
                   X = 1e-3/2,
                   Y = 0.8528e-3/2,
                   Z = 10e-3,
                   laser_setup = six_dirs,
                  )

UW_Trap = Ion_Trap(w_E = 2 * np.pi * 18.2e6,
                   V_0 = 1000,
                   V_caps = 10,
                   X = 0.930e-3,
                   Y = 0.793e-3,
                   Z = 2.84e-3,
                   laser_setup = six_dirs,
                  )