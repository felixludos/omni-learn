import numpy as np
from collections import namedtuple
import itertools as it
from lab_setup import *
# Forces

class E_Field:
    def __init__(self, env):
        self.name = 'Electric Field'
        self.E_field_caps_const = env.charges * np.array([1.,1.,-2]) * env.trap.V_caps / env.trap.Z**2
        self.E_field_const = env.charges * np.array([1./env.trap.X**2, -1./env.trap.Y**2, 0]) * env.trap.V_0 
        
    def calc_force(self, env): # calc force
        self.force = (self.E_field_const * np.cos(env.trap.w_E * env.time) + self.E_field_caps_const) * env.pos
        return self.force

    def calc_energy(self, env):
        self.energy = - np.sum((self.E_field_const * np.cos(env.trap.w_E * env.time) + self.E_field_caps_const) * env.pos**2)
        
class Coulomb_Repulsion:
    def __init__(self, env):
        self.name = 'Coulomb Repulsion'
        self.exclude_me = np.identity(env.num_particles) == 0
        self.Repulsion_consts = [K * q * env.charges[remove,:] for q, remove in zip(env.charges, self.exclude_me)]
        self.force = np.zeros((env.num_particles,3))
        self.energy = 0
        
    def calc_force(self, env):  # calc force
        if env.num_particles == 1:
            return self.force
        self.force = []
        for r, const, remove in zip(env.pos, self.Repulsion_consts, self.exclude_me):
            d = r - env.pos[remove,:]
            dist = np.sum(d ** 2, axis=1) ** 1.5
            self.force.append(np.sum(const * d / dist[:,np.newaxis], axis=0))
        self.force = np.vstack(self.force)
        return self.force
        
    def calc_energy(self, env):
        self.energy = 0
        for r, const, remove in zip(env.pos, self.Repulsion_consts, self.exclude_me):
            dist = np.sum((r - env.pos[remove,:]) ** 2, axis=1) ** 0.5
            #print dist.shape
            self.energy += np.sum(const.T / dist)

def calc_scattering_rate(speed, const, intensity_ratio, detuning, cooling_k, line_width):
    return const / (1 + intensity_ratio + 4 * ((detuning + cooling_k * speed)/line_width)**2)
    
def de_Broglie(l):
    return h / l

def resample_selected(x):
    return np.random.rand() if x == 1 else x
    
class Laser_Cooling: # includes radiative heating
    def __init__(self, env):
        self.name = 'Laser Cooling'
        #self.div = lambda l: h / l
        env.photon_momenta = np.vectorize(de_Broglie)(env.cooling_laser_wavelengths).reshape(env.num_particles, 1)
        self.detunings = env.natural_line_widths / 2
        self.samples = np.random.rand(env.num_particles,env.trap.laser_setup.shape[0])
        self.last_event = np.zeros((env.num_particles,env.trap.laser_setup.shape[0]))
        self.scatter_rate_constants = env.natural_line_widths * env.intensity_ratios / 2
        #self.calc_scatter_rate = lambda i, speed: self.scatter_rate_constants[i,0] / (1 + env.intensity_ratios[i,0] + 4 * ((self.detunings[i,0] + env.cooling_laser_ks[i,0]*speed)/env.natural_line_widths[i,0])**2)
        #self.resample_fn = lambda x: np.random.rand() if x == 1 else x
        self.resample = np.vectorize(resample_selected)
        self.pushes = np.dstack([env.trap.laser_setup]*env.num_particles).T
        env.photon_events = np.zeros((env.num_particles,env.trap.laser_setup.shape[0]))
        self.no_cooling = not env.is_cooling.any()
        if self.no_cooling:
            self.force = np.zeros((env.num_particles,3))
        self.energy = 0
        
    def calc_force(self, env):
        if self.no_cooling:
            return self.force
        force = []
        for ion, v in enumerate(env.vel):
            if env.is_cooling[ion]:
                speed = env.trap.laser_setup.dot(v)
                rates = calc_scattering_rate(speed, self.scatter_rate_constants[ion,:], env.intensity_ratios[ion,:], self.detunings[ion,:], env.cooling_laser_ks[ion,:], env.natural_line_widths[ion,:]) 
                #rates = self.calc_scatter_rate(ion, p)
                probs = 1. - np.exp(- rates * (env.time - self.last_event[ion,:]))
                events = probs - self.samples[ion,:] > 0
                occurred = np.sum(events)
                env.photon_events[ion,:] += events
                if occurred:
                    self.last_event[ion,events] = env.time
                    self.samples[ion, events] = 1
                    self.samples[ion,:] = self.resample(self.samples[ion,:])
                    force.append(env.photon_momenta[ion,0] # magnitude
                                 * (np.sum(env.trap.laser_setup[events,:],axis=0) # absorption
                                    + gen_unit_vector(sum_of=occurred)) # emission
                                 / env.timestep) # momentum to impulse
                else:
                    force.append(zero_vector)
                afterwards = env.trap.laser_setup.dot(v+force[-1]*env.timestep/env.masses[ion,0])
            else:
                force.append(zero_vector)
        self.force = np.vstack(force)
        return self.force
        
    def calc_energy(self,env):
        return

class Laser_Cooling_Old: # includes radiative heating
    def __init__(self, env):
        self.name = 'Laser Cooling'
        env.photon_momenta = np.vectorize(lambda l: h / l)(env.cooling_laser_wavelengths).reshape(env.num_particles, 1)
        self.detunings = env.natural_line_widths / 2
        self.samples = np.random.rand(env.num_particles,env.trap.laser_setup.shape[0])
        self.last_event = np.zeros((env.num_particles,env.trap.laser_setup.shape[0]))
        self.scatter_rate_constants = env.natural_line_widths * env.intensity_ratios / 2
        self.calc_scatter_rate = lambda i, speed: self.scatter_rate_constants[i,0] / (1 + env.intensity_ratios[i,0] + 4 * ((self.detunings[i,0] + env.cooling_laser_ks[i,0]*speed)/env.natural_line_widths[i,0])**2)
        self.resample = np.vectorize(lambda x: np.random.rand() if x == 1 else x)
        self.pushes = np.dstack([env.trap.laser_setup]*env.num_particles).T
        env.photon_events = 0
        self.no_cooling = not env.is_cooling.any()
        if self.no_cooling:
            self.force = np.zeros((env.num_particles,3))
        self.energy = 0
        print('samples', self.samples)
        
    def calc_force(self, env):
        if self.no_cooling:
            return
        force = []
        for ion, v in enumerate(env.vel):
            if env.is_cooling[ion]:
                force.append(np.array([0.,0.,0.]))
                for j, laser in enumerate(env.trap.laser_setup):
                    #print env.time / env.timestep, '--', j
                    p = laser.dot(v)
                    #print '\tpointing', p
                    rate = self.calc_scatter_rate(ion, p)
                    prob = 1. - np.exp(- rate * (env.time - self.last_event[ion,j]))
                    #print '\trate', rate
                    #print '\tprob', prob 
                    #print '\tsample', self.samples[ion,j]
                    if prob - self.samples[ion,j] > 0:
                        #print '\t\t<bing>'
                        env.photon_events += 1
                        self.samples[ion, j] = np.random.rand()
                        self.last_event[ion,j] = env.time 
                        magnitude = env.photon_momenta[ion,0]
                        absorption = laser
                        emission = gen_unit_vector()
                        f = magnitude * (absorption + emission) / env.timestep
                        #print '\t\tabs,em,f', [np.sqrt(np.sum(l**2)) for l in [absorption, emission, absorption+emission]]
                        #print '\t\ttogether', absorption.dot(absorption+emission)
                        #print np.vstack([absorption, emission, f])
                        force[-1] += f
                    #print '\t\tapplied', new_f
                    #print '\tafterward', laser.dot(v+force[-1]*env.timestep/env.masses[ion,0])

            else:
                force.append(zero_vector)
        self.force = np.vstack(force)
        
    def calc_energy(self,env):
        return
    
class Laser_Cooling_Fast:
    def __init__(self, env):
        self.name = 'Laser Cooling'
        env.photon_momenta = np.vectorize(lambda l: h / l)(env.cooling_laser_wavelengths).reshape(env.num_particles, 1)
        self.detunings = env.natural_line_widths / 2
        self.samples = np.random.rand(env.num_particles,env.trap.laser_setup.shape[0])
        self.last_event = np.zeros((env.num_particles,env.trap.laser_setup.shape[0]))
        self.scatter_rate_constants = env.natural_line_widths * env.intensity_ratios / 2
        self.calc_scatter_rate = lambda i, vel: self.scatter_rate_constants[i,0] / (1 + env.intensity_ratios[i,0] + 4 * ((self.detunings[i,0] + env.cooling_laser_ks[i,0]*vel)/env.natural_line_widths[i,0])**2)
        self.resample = np.vectorize(lambda x: np.random.rand() if x == 1 else x)
        self.pushes = np.dstack([env.trap.laser_setup]*env.num_particles).T
        env.photon_events = 0
        self.no_cooling = not env.is_cooling.any()
        if self.no_cooling:
            self.force = np.zeros((env.num_particles,3))
        self.energy = 0
    
    def calc_energy(self,env):
        return
    
    def calc_force(self, env):
        rates = self.calc_scatter_rates(np.dot(env.vel, env.trap.laser_setup.T))
        probs = 1. - np.exp(- rates * (env.time - self.last_event))
        p = np.zeros(env.vel.shape)
        events = prob - self.samples > 0
        env.photon_events += np.sum(events)
        self.samples[events] = 1
        self.samples = self.resample(self.samples)
        absorption = np.vstack([np.sum(op[:,px],axis=1) for px,op in zip(picks,options)])
        emission = np.vstack([gen_unit_vector(sum_of=i) if i > 0 else zero_vector for i in np.sum(events,axis=1)])
        self.force = env.photon_momentum * (absorption + emission) / env.timestep
        