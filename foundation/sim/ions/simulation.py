import os
import numpy as np
from collections import deque
import pickle
import multiprocessing as mp
#import matplotlib
#matplotlib.use('Qt5Agg')
#%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import NavigationToolbar2, Event
#from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.special import digamma
from stats import *
from forces import *
from lab_setup import *

def sample_distribution(distr, samples, arange):

        distr = np.vectorize(distr)

        probs = distr(arange)

        probs /= np.sum(probs)

        cdf = np.cumsum(probs)

        picks = np.random.rand(samples)

        return np.array([ arange[cdf < s][-1] for s in picks ])

    
def eq_distance(n, trap): # for n +1 ions in trap
    def N(n): # for coulomb calculation
        return 1 + n * ( -digamma(1) + digamma(n) - 1)

    def M(n): # for electric field calculation
        if n % 2 == 1: # n is odd
            return (n-1)*n*(n+1)
        return n**3-n-6
    if n == 1:
        return 0.
    if n == 2:
        return (2 * K * e * trap.Z**2 / trap.V_caps)**(1./3)
    const = 6 * K * e * trap.Z**2 / (trap.V_caps)
    combo = N(n) / M(n)
    return (const * combo)**(1./3)

def make_maxwell_speed_distr(m, T_0):
    return lambda v: np.sqrt((m / (2 * np.pi * kB * T_0)) ** 3) * 4 * np.pi * v ** 2 * np.exp(- m * v ** 2 / (2 * kB * T_0))

cooling_limit = h * Barium.natural_line_width / 4 / np.pi / kB
def temp_setting(env):
    sigma = 0.0001
    T_0 = cooling_limit
    v_0 = np.sqrt(T_0 * 3. * kB / env.masses)
    distribs = [make_maxwell_speed_distr(m,T_0) for m in env.masses]
    speeds = np.array([sample_distribution(distr, 1, np.arange(0, 5 * v, 0.001)) for distr, v in zip(distribs, v_0)]).reshape(env.num_particles,1)
    
    vel = np.vstack([gen_unit_vector() for _ in range(env.num_particles)])
    vel *= speeds
    
    #if equilibrium_pos:
    dist = eq_distance(env.num_particles, env.trap)
        
    if env.num_particles == 1:
        pos = np.array([[0.,0.,0.]])
    elif env.num_particles == 2:
        pos = np.array([[0., 0., dist/2], [0., 0., -dist/2]])
    else:
        pos = np.array([[0., 0., dist*i] for i in range(env.num_particles)])
        max_pos = dist*(env.num_particles-1)
        pos[:,2] -= max_pos / 2
    #else:
    #    pos = np.hstack([np.random.randn(env.num_particles,1) * sigma * l for l in [env.trap.X, env.trap.Y, env.trap.Z]]) 

    return pos, vel

def mp_run_job(func, arg, q):
    q.put(func(arg))

def mp_run(jobs, arg): # assumes order of outputs doesn't matter, and input to each job is the same
    procs = []
    q = mp.Queue()
    outputs = []
    for job in jobs:
        procs.append(mp.Process(target=mp_run_job, args=(job, arg, q)))
        procs[-1].start()
    for proc in procs:
        proc.join()
    while not q.empty():
        outputs.append(q.get())
    return outputs

class Simulation_Environment:
    def __init__(self, trap, atoms, forces, timestep, init_setting):
        
        # simulation contents
        self.trap = trap
        self.atoms = atoms
        
        self.timestep = timestep
        self.time = 0
        
        # calc constants
        self.num_particles = len(self.atoms)
        self.symbols = np.array([a.symbol for a in self.atoms]).reshape(self.num_particles,1)
        self.masses = np.array([a.mass for a in self.atoms]).reshape(self.num_particles,1)
        self.charges = np.array([a.charge for a in self.atoms]).reshape(self.num_particles,1)
        self.natural_line_widths = np.array([a.natural_line_width for a in self.atoms]).reshape(self.num_particles,1)
        self.rabi_frequencies = np.array([a.rabi_frequency for a in self.atoms]).reshape(self.num_particles,1)
        self.intensity_ratios = 2 * self.rabi_frequencies**2 / self.natural_line_widths**2
        self.cooling_laser_wavelengths = np.array([a.cooling_laser_wavelength for a in self.atoms]).reshape(self.num_particles,1)
        self.cooling_laser_ks = 2 * np.pi / self.cooling_laser_wavelengths
        self.is_cooling = np.array([a.cooling for a in self.atoms]).reshape(self.num_particles,1)
        
        # make forces
        self.forces = [f(self) for f in forces]
        self.force_funcs = [action.calc_force for action in self.forces]
        
        # init pos, vel, acc
        self.default_setting = init_setting
        self.reset_kinematics()
        
    def reset_kinematics(self, setting=None):
        if setting is None:
            setting = self.default_setting
        self.pos, self.vel = setting(self)
        self.acc = np.zeros((self.num_particles, 3))
    
    def calc_force(self):
        forces = [force_func(self) for force_func in self.force_funcs]
        return sum(forces)
    
    def calc_force_parallel(self):
        return sum(mp_run(self.force_funcs, self, len(self.force_funcs)))
        #return sum(self.force_pool.map(exec_func, [lambda: action.calc_force(self) for action in self.forces]))
    
    def step(self, n_steps=1, parallel=False):
        get_forces = self.calc_force_parallel if parallel else self.calc_force
        for _ in range(n_steps):
            # single verlet step
            half_vel = self.vel + self.acc * self.timestep / 2
    
            new_pos = self.pos + half_vel * self.timestep
        
            new_acc = get_forces() / self.masses
            
            print(self.time)
            print(new_acc)

            new_vel = half_vel + new_acc * self.timestep / 2

            self.pos, self.vel, self.acc = new_pos, new_vel, new_acc
            self.time += self.timestep
            


class Trajectory:
    def __init__(self, name, sim_parameters, stats, print_step=1, print_time=None, seed=None):
        assert print_step is not None or print_time is not None, 'Either print_step or print_time must be specified'
        if print_time is not None:
            print_step = int(np.round(print_time/sim.timestep))
        assert print_step >= 1, 'print_step is too small, increase print_step or print_time'
        if seed is None:
            self.seed = np.random.randint(1000000) # TODO: pick from all ints
        np.random.seed(self.seed)
        self.print_step = print_step
        self.name = name
        self.sim = Simulation_Environment(**sim_parameters)
        self.stats = {stat.name:stat for stat in stats}
        self.reset()
        
    def reset(self):
        self.sim.reset_kinematics()
        self.pos = []
        self.vel = []
        self.times = []
        for stat in self.stats.values():
            stat.reset(self.sim)
        
    def run(self, frames, print_progress=True):
        
        use_int = frames / self.print_step == 100
        
        for frame in range(0, frames, self.print_step):
            # print progress
            if print_progress:
                print(int(np.round(float(frame) / frames * 100)+1) if use_int else np.round(float(frame) / frames * 100,3),)

            self.times.append(self.sim.time)
            self.pos.append(self.sim.pos)
            self.vel.append(self.sim.vel)
            for stat in self.stats.values():
                stat.calc(self.sim)

            self.sim.step(n_steps=self.print_step)

    def calc_pos_means(self, axis=2): # z by default

        coords = np.vstack([r[:,axis] for r in self.pos])

        return coords.mean(0)


    def plot_stats(self, *stat_names):
        
        if len(stat_names) == 0:
            stat_names = self.stats.keys()
            
        axes = []
            
        for stat in [self.stats[name] for name in stat_names]:
            fig, ax = plt.subplots()
            
            axes.append(ax)

            data = stat.get_data()

            if stat.labels is not None:
                for line, name in zip(data, stat.labels):
                    ax.plot(self.times, line, label=name)
                ax.legend()
            else:
                for line in data:
                    ax.plot(self.times, line)

            if stat.level is not None:
                ax.plot([self.times[0],self.times[-1]], [stat.level, stat.level],'r--')
                
            ax.set_xlabel('Time (s)')
            if stat.y_axis is not None:
                ax.set_ylabel(stat.y_axis)
            if stat.title is not None:
                ax.set_title(stat.title)
        
        return axes

    def plot_traj(self, animated=False, figax=None):
        if figax is not None:
            fig, ax = figax
        else:
            fig = plt.figure()
            ax = p3.Axes3D(fig)
            #ax = fig.gca(projection='3d')
        if animated:
            return self._plot_animated(fig, ax)
        for r, v in zip(self.pos, self.vel):
            arrows = np.vstack([r.T, v.T*self.sim.timestep*self.print_step])
            ax.quiver(*arrows, pivot='tail',arrow_length_ratio=0)
            ax.plot(*(r.T),ls='',marker='.',markersize=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return ax
    
    def _plot_animated(sim_self, fig, ax):
        '''
        forward = NavigationToolbar2.forward
        def new_forward(self, *args, **kwargs):
            s = 'forward_event'
            event = Event(s, self)
            #event.foo = 100
            self.canvas.callbacks.process(s, event)
        NavigationToolbar2.forward = new_forward

        back = NavigationToolbar2.back
        def new_backward(self, *args, **kwargs):
            s = 'back_event'
            event = Event(s, self)
            #event.foo = 100
            self.canvas.callbacks.process(s, event)
        NavigationToolbar2.back = new_backward
        '''
        
        class Plotter:
            def __init__(self, data, times):
                self.times = times
                self.data = data # 3d: (ion, pos(xyz)/vel(xyz), traj) - (N, 6, traj)
                self.colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
                self.colors = (self.data.shape[0] / len(self.colors) + 1) * self.colors
                self.frame = 0
                self.fig = fig
                self.ax = ax
                self.ax.set_xlim((self.data[:,0,:].min(),self.data[:,0,:].max()))
                self.ax.set_ylim((self.data[:,1,:].min(),self.data[:,1,:].max()))
                self.ax.set_zlim((self.data[:,2,:].min(),self.data[:,2,:].max()))
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                self.ax.set_zlabel('Z')
                
                self.artists = [self.ax.text(0, 0,0, '')] #[self.ax.set_title('0', animated=True)]
                #self.artists += [self.ax.plot(*ion[:3,:1], ls='', marker='.')[0] for ion in self.data]
                #for ion in self.data:
                #    self.ax.plot(*ion[:3], ls='', marker='.')
                #    self.ax.quiver(*ion, pivot='tail', color='g',arrow_length_ratio=0)
                
                self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.data.shape[-1], interval=500, blit=False, repeat=True)
                self.paused = False
                
                #return
                
                #self.fig.canvas.mpl_connect('forward_event', self.forward)
                #self.fig.canvas.mpl_connect('back_event', self.back)
                #self.fig.canvas.mpl_connect('home_event', self.play)
                
                #print 'events added'
                
                
            def update(self, i):
                self.draw()
                self.frame += 1
                return self.artists
                
            def draw(self):
                #self.ax.cla()
                if self.frame > self.data.shape[-1]:
                    self.frame = 0
                    self.ax.cla()
                #    pass # clear ax with ax.cla()?
                if self.frame < 0:
                    self.frame = self.data.shape[-1] - 1

                #print 'frame', self.frame
                #self.ax.set_title(str(self.frame))

                self.artists[0].set_text('T = '+str(self.times[self.frame])+' s')

                #for ion, ion_line in enumerate(self.artists[1:]):
                for ion, ion_color in zip(self.data,self.colors):
                    #ion_line.set_data(self.data[ion,:2,:self.frame])
                    #ion_line.set_3d_properties(self.data[ion,2,:self.frame])
                    self.ax.plot(*ion[:3,self.frame:self.frame+1], color=ion_color, ls='', marker='.')
                    self.ax.quiver(*ion[:,self.frame:self.frame+1], pivot='tail', color=ion_color,arrow_length_ratio=0)

                #return self.artists
                #self.ax.quiver(*self.data[self.frame], pivot='tail', arrow_length_ratio=0)
                
            def play(self, *args, **kwargs):
                print('pressed home')
                self.paused ^= True
                #if self.paused:
                #    self.ani.event_source.stop()
                #else:
                #    self.ani.event_source.start()
            def forward(self, *args, **kwargs):
                print('pressed forward')
                self.frame += 1
                self.draw()
            def back(self,*args, **kwargs):
                print('pressed back')
                self.frame -= 1
                self.draw()
        start_points = np.stack(sim_self.pos)
        vel_points = np.stack(sim_self.vel)*sim_self.sim.timestep*sim_self.print_step
        plotter = Plotter(np.transpose(np.concatenate([start_points.T, vel_points.T],axis=0),(1,0,2)), times=sim_self.times)
        NavigationToolbar2.home = plotter.play
        NavigationToolbar2.forward = plotter.forward
        NavigationToolbar2.back = plotter.back
        return plotter

    def plot_motion(self, axis, show_pos=False, separate_plots=False, pos_scaling=1e-6, freq_scaling=1e3, print_max=True, save=False):
        axis_names = {0:'X', 1:'Y', 2:'Z'}
        pos_scalings = {1:'m', 1e-3:'mm', 1e-4:'100 um', 1e-5:'10 um', 1e-6:'um', 1e-7:'100 nm', 1e-8:'10 nm', 1e-9:'nm'}
        freq_scalings = {1:'Hz', 1e3:'kHz', 1e4:'10 kHz', 1e5:'100 kHz', 1e6:'MHz', 1e7:'10 MHz', 1e8:'100 MHz', 1e9:'GHz'}

        coords = np.vstack([r[:,axis] for r in self.pos]).T
        fs = coords.shape[1] / (self.times[-1] - self.times[0]) # sample rate
        if print_max: # TODO: print all peaks - using peak finder
            print('Max at (' + freq_scalings[freq_scaling] + '):')
        
        if show_pos:
            pos_fig, pos_ax = plt.subplots()
        
        if separate_plots:
            freq_fig, axes = plt.subplots(len(coords), sharex=True)
            if len(coords) == 1:
                axes = (axes,)
            colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
            
            for ion, (x,freq_ax, color) in enumerate(zip(coords,axes, colors)):
                x /= pos_scaling
                spec = np.abs(np.fft.fft(x-np.mean(x)))
                spec = spec[:len(spec)/2]
                spec /= spec.max() # normalize principle peak to 1
                freqs = np.arange(len(spec))*fs/len(x)/freq_scaling
                if show_pos:
                    pos_ax.plot(self.times,x, color=color)
                freq_ax.plot(freqs,spec, color=color)
                if print_max:
                    print('\t ion', ion, freqs[spec.argmax()])
                    
                if save:
                    try:
                        os.mkdir(self.name)
                    except:
                        pass
                    if show_pos:
                        np.save(self.name + '/' + axis_names[axis] + '_motion_ion' + str(ion), np.vstack([self.times, x]).T)
                    np.save(self.name + '/' + axis_names[axis] + '_spec_ion' + str(ion), np.vstack([freqs, spec]).T)

            axes[-1].set_xlabel('Frequency ('+ freq_scalings[freq_scaling] +')')
            axes[len(axes)/2].set_ylabel('Magnitude')
            axes[0].set_title('Ion ' + axis_names[axis] + ' Axis Frequency Spectrum')
            
            if save:
                for ax in axes:
                    ax.set_xlim((0,2e3))
                    ax.set_ylim((-0.02, 1.02))
                freq_fig.savefig(self.name + '/freq_'+axis_names[axis]+'.png')
                if show_pos:
                    pos_fig.savefig(self.name + '/pos_'+axis_names[axis]+'.png')
                    
                print('saved to', self.name + '/')
            
            return
        
        #print coords.shape
        
        freq_fig, freq_ax = plt.subplots()
        
        for ion, x in enumerate(coords):
            x /= pos_scaling
            spec = np.abs(np.fft.fft(x-np.mean(x)))
            spec = spec[:len(spec)/2]
            freqs = np.arange(len(spec))*fs/len(x)/freq_scaling
            if show_pos:
                pos_ax.plot(self.times,x)
            freq_ax.plot(freqs,spec)
            if print_max:
                print('\t ion', ion, freqs[spec.argmax()])

        freq_ax.set_xlabel('Frequency ('+ freq_scalings[freq_scaling] +')')
        freq_ax.set_ylabel('Magnitude')
        freq_ax.set_title('Ion ' + axis_names[axis] + ' Axis Frequency Spectrum')
        if show_pos:
            pos_ax.set_xlabel('Time (s)')
            pos_ax.set_ylabel('Position ('+ pos_scalings[pos_scaling] +')')
            pos_ax.set_title('Ion ' + axis_names[axis] + ' Axis Positions')
            return pos_ax, freq_ax
        return freq_ax

    def save_state(self, root=''):
        pickle.dump(self, open(root + self.name + '.state','w'))
        print('Sim State saved to:', root + self.name)
        
    def load_state(self, filename):
        self = pickle.load(open(filename, 'r'))
        print('Sim State loaded from:', filename)
    
    def save_xyz_file(self, root='', scaling=1e6): # TODO: include stats in comment line
        with open(root + self.name + '.xyz', 'w') as f:
            for t, p in zip(self.times, self.pos):
                f.write(str(self.sim.num_particles) + '\n')
                f.write('Time: ' + str(t) + ' s\n')
                for sym, (x,y,z) in zip(self.sim.symbols,p):
                    f.write(sym[0] + ' ' + str(x*scaling) + ' ' + str(y*scaling) + ' ' + str(z*scaling) + '\n')
        print(root + self.name + '.xyz', 'saved')


