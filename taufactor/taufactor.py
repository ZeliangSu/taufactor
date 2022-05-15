"""Main module."""
import numpy as np
try: 
    import cupy as cp
except ModuleNotFoundError as e:
    import numpy as cp

from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm import tqdm

class Solver:
    """
    Default solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """
    def __init__(self, img, precision=cp.single, bc=(-0.5, 0.5), D_0=1):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        """
        # add batch dim now for consistency
        self.D_0 = D_0
        self.top_bc, self.bot_bc = bc
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.cpu_img = img
        self.precision = precision
        # VF calc
        self.VF = np.mean(img)
        # save original image in cuda
        img = cp.array(img, dtype=self.precision)
        self.ph_bot = cp.sum(img[:, -1]) * self.bot_bc
        self.ph_top = cp.sum(img[:, 0]) * self.top_bc

        # init conc
        self.conc = self.init_conc(img)

        # create nn map
        self.nn = self.init_nn(img)

        #checkerboarding
        self.w = 2 - cp.pi / (1.5 * img.shape[1]) #todo: why img.shape[1]?
        self.cb = self.init_cb(img)

        # solving params
        bs, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)
        self.converged = False
        self.semi_converged = False
        self.iter=0
        img = None

        # Results
        self.tau=None
        self.D_eff=None
        self.D_mean=None

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x * 2)
        vec = cp.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec.astype(self.precision)

        return self.pad(img * vec, [self.top_bc * 2, self.bot_bc * 2]) #todo: init for the phase == 1

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim) #todo: more degree of liberty, higher is the value in nn
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        return nn

    def init_cb(self, img):
        bs, x, y, z = img.shape
        cb = np.zeros([x, y, z])
        a, b, c = np.meshgrid(range(x), range(y), range(z), indexing='ij')
        cb[(a + b + c) % 2 == 0] = 1
        cb *= self.w
        return [cp.roll(cp.array(cb), sh, 0) for sh in [0, 1]]

    def pad(self, img, vals=[0] * 6):
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[:2] = (0, 0)
        img = cp.pad(img, np.reshape(to_pad, (4, 2)), 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def crop(self, img, c = 1):
        return img[:, c:-c, c:-c, c:-c]

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            # note: out is the average concentration matrix of its surroundings (excluded solid voxels)
            out = self.conc[:, 2:, 1:-1, 1:-1] + \
                  self.conc[:, :-2, 1:-1, 1:-1] + \
                  self.conc[:, 1:-1, 2:, 1:-1] + \
                  self.conc[:, 1:-1, :-2, 1:-1] + \
                  self.conc[:, 1:-1, 1:-1, 2:] + \
                  self.conc[:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 20 == 0:
                lt = abs(cp.sum(out[:, 0]) - self.ph_top)
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.converged, D_rel = self.check_convergence(lt, lb, verbose, conv_crit, start, iter_limit)
                print(D_rel)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter%2] # note: interesting!
            self.conc[:, 1:-1, 1:-1, 1:-1] += out
            self.iter += 1
        self.D_mean = self.D_0
        self.tau=self.VF/D_rel if D_rel != 0 else cp.inf
        self.D_eff=self.D_mean*D_rel
        self.end_simulation(iter_limit, verbose, start)
        return self.tau

    def check_convergence(self, lt, lb, verbose, conv_crit, start, iter_limit):
        loss = lt - lb
        # print progress
        if self.iter % 100 == 0:
            if verbose == 'per_iter':
                print(self.iter, abs(loss))

        # check for convergence if loss is good
        if abs(loss) < conv_crit * 0.01:
            if self.semi_converged:
                cvf = self.check_vertical_flux(conv_crit)
                if cvf:
                    if cvf == 'zero_flux':
                        return True, 0
                    #note: get() moves array from GPU to host, equivalent to cp.asnumpy()
                    try:
                        return True, ((lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc)).get()
                    except AttributeError as e:
                        return True, (lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc) 
            else:
                self.semi_converged = True
        else:
            self.semi_converged = False

        # increase precision to double if currently single
        if self.iter >= iter_limit:
            if self.precision == cp.single:
                print('increasing precision to double')
                self.iter = 0
                self.conc = cp.array(self.conc, dtype=cp.double)
                self.nn = cp.array(self.nn, dtype=cp.double)
                self.precision = cp.double
            else:
                print('Did not converge in the iteration limit')
                try:
                    return True, ((lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc)).get()
                except AttributeError as e:
                    return True, (lt + lb) * self.L_A / abs(self.top_bc - self.bot_bc)

        return False, False

    def check_vertical_flux(self, conv_crit):
        vert_flux = self.conc[:, 1:-1, 1:-1, 1:-1] - self.conc[:, :-2, 1:-1, 1:-1]
        vert_flux[self.conc[:, :-2, 1:-1, 1:-1] == 0] = 0
        vert_flux[self.conc[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))[1:-1]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or np.isnan(err).item():
            return True
        if fl.min() == 0:
            return 'zero_flux'
        return False

    def conc_map(self, lay=0):
        """
        Plots a concentration map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D conc map
        """
        try: 
            img = self.conc[0, 1:-1, 1:-1, 1:-1].get()
        except AttributeError as e:
            img = self.conc[0, 1:-1, 1:-1, 1:-1]

        img[self.cpu_img[0, :, :, :] == 0] = -1
        try:
            plt.imshow(img[:, :, lay])
        except TypeError as e:
            print('*** Take the real part')
            plt.imshow(img[:, :, lay].real())
        return img

    def flux_map(self, lay=0):
        """
        Plots a flux map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D flux map
        """
        flux = cp.zeros_like(self.conc)
        ph_map = self.pad(cp.array(self.cpu_img))
        for dim in range(1, 4):
            for dr in [1, -1]:
                flux += abs(cp.roll(self.conc, dr, dim) - self.conc) * cp.roll(ph_map, dr, dim)
        try:
            flux = flux[0, 2:-2, 1:-1, 1:-1].get()
        except AttributeError as e:
            flux = flux[0, 2:-2, 1:-1, 1:-1]
        flux[self.cpu_img[0, 1:-1] == 0] = 0

        try:
            plt.imshow(flux[:, :, lay])
        except TypeError as e:
            print('*** Take the real part')
            plt.imshow(flux[:, :, lay].real())
        return flux

    def end_simulation(self, iter_limit, verbose, start):
        if self.iter==iter_limit -1:
            print('Warning: not converged')
            converged = 'unconverged value of tau'
        converged = 'converged to'
        if verbose:
            print(f'{converged}: {self.tau} \
                  after: {self.iter} iterations in: {np.around(timer() - start, 4)}  \
                  seconds at a rate of {np.around((timer() - start)/self.iter, 4)} s/iter')
        self.converged = False
        self.semi_converged = False
        self.iter = 0

class PeriodicSolver(Solver):
    """
    Periodic Solver (works for non-periodic structures, but has higher RAM requirements)
    Once solve method is called, tau, D_eff and D_rel are available as attributes.
    """
    def __init__(self, img, precision=cp.single, bc=(-0.5, 0.5), D_0=1):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with 1s conductive and 0s non-conductive
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity

        """
        super().__init__(img, precision, bc, D_0)
        self.conc = self.pad(self.conc)[:, :, 2:-2, 2:-2]

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [2, 2]))[:, :, 2:-2, 2:-2]
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim)
        # avoid div 0 errors
        nn = nn[:, 2:-2]
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        return nn

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, D_0=1):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            out = cp.zeros_like(self.conc)
            for dim in range(1, 4):
                for dr in [1, -1]:
                    out += cp.roll(self.conc, dr, dim)
            out = out[:, 2:-2]
            out /= self.nn
            if self.iter % 50 == 0:
                lt = abs(cp.sum(out[:, 0]) - self.ph_top)
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.converged, D_rel = self.check_convergence(lt, lb, verbose, conv_crit, start, iter_limit)
            out -= self.conc[:, 2:-2]
            out *= self.cb[self.iter % 2]
            self.conc[:, 2:-2] += out
            self.iter += 1

        self.D_mean=D_0
        self.tau = self.VF/D_rel if D_rel !=0 else cp.inf
        self.D_eff=D_0*D_rel
        self.end_simulation(iter_limit, verbose, start)
        return self.tau

    def check_vertical_flux(self, conv_crit):
        vert_flux = abs(self.conc - cp.roll(self.conc, 1, 1))
        vert_flux[self.conc == 0] = 0
        vert_flux[cp.roll(self.conc, 1, 1) == 0] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))[3:-2]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or np.isnan(err):
            return True
        if fl.min() == 0:
            return 'zero_flux'

    def flux_map(self, lay=0):
        """
        Plots a flux map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D flux map
        """
        flux = cp.zeros_like(self.conc)
        ph_map = self.pad(self.pad(cp.array(self.cpu_img)))[:, :, 2:-2, 2:-2]
        for dim in range(1, 4):
            for dr in [1, -1]:
                flux += abs(cp.roll(self.conc, dr, dim) - self.conc) * cp.roll(ph_map, dr, dim)
        try:
            flux = flux[0, 2:-2].get()
        except AttributeError as e:
            flux = flux[0, 2:-2]
        flux[self.cpu_img[0] == 0] = 0
        plt.imshow(flux[:, :, lay])
        return flux

    def conc_map(self, lay=0):
        """
        Plots a concentration map perpendicular to the direction of flow
        :param lay: depth to plot
        :return: 3D conc map
        """
        try:
            img = self.conc[0, 2:-2].get()
        except AttributeError as e:
            img = self.conc[0, 2:-2]
        img[self.cpu_img[0] == 0] = -1
        plt.imshow(img[:, :, lay])
        plt.show()

class MultiPhaseSolver(Solver):
    """
    Multi=phase solver for two phase images. Once solve method is
    called, tau, D_eff and D_rel are available as attributes.
    """
    def __init__(self, img, cond={1:1}, precision=cp.single, bc=(-0.5, 0.5)):
        """
        Initialise parameters, conc map and other tools that can be re-used
        for multiple solves.

        :param img: input img with n conductive phases labelled as integers, and 0s for non-conductive
        :param cond: dict with n phase labels as keys, and their corresponding conductivities as values e.g
        for a 2 phase material, {1:0.543, 2: 0.420}, with 1s and 2s in the input img
        :param precision:  cp.single or cp.double
        :param bc: Upper and lower boundary conditions. Leave as default.
        :param D_0: reference material diffusivity
        """

        if 0 in cond.values():
            raise ValueError('0 conductivity phase: non-conductive phase should be labelled 0 in the input image and ommitted from the cond argument')
        self.cond = {ph: 0.5 / c for ph, c in cond.items()}
        super().__init__(img, precision, bc)
        self.pre_factors = self.nn[1:]
        self.nn = self.nn[0]
        self.VF = {p:np.mean(img==p) for p in np.unique(img)}

    def init_nn(self, img):
        #conductivity map
        img2 = cp.zeros_like(img)
        for ph in self.cond:
            c = self.cond[ph]
            img2[img == ph] = c
        img2 = self.pad(self.pad(img2))
        img2[:, 1] = img2[:, 2]
        img2[:, -2] = img2[:, -3]
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        nn_list = []
        for dim in range(1, 4):
            for dr in [1, -1]:
                shift = cp.roll(img2, dr, dim)
                sum = img2 + shift
                sum[shift==0] = 0
                sum[img2==0] = 0
                sum = 1/sum
                sum[sum == cp.inf] = 0
                nn += sum
                nn_list.append(self.crop(sum, 1))
        # remove the two paddings
        nn = self.crop(nn, 2)
        # avoid div 0 errors
        nn[img == 0] = cp.inf
        nn[nn == 0] = cp.inf
        nn_list.insert(0, nn)
        return nn_list

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x + 1)
        vec = cp.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec.astype(self.precision)
        img1 = cp.array(img)
        img1[img1 > 1] = 1
        return self.pad(img1 * vec, [self.top_bc, self.bot_bc])

    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """

        start = timer()
        while not self.converged:
            self.iter += 1
            out = self.conc[:, 2:, 1:-1, 1:-1] * self.pre_factors[0][:, 2:, 1:-1, 1:-1] + \
                  self.conc[:, :-2, 1:-1, 1:-1] * self.pre_factors[1][:, :-2, 1:-1, 1:-1] + \
                  self.conc[:, 1:-1, 2:, 1:-1] * self.pre_factors[2][:, 1:-1, 2:, 1:-1] + \
                  self.conc[:, 1:-1, :-2, 1:-1] * self.pre_factors[3][:, 1:-1, :-2, 1:-1] + \
                  self.conc[:, 1:-1, 1:-1, 2:] * self.pre_factors[4][:, 1:-1, 1:-1, 2:] + \
                  self.conc[:, 1:-1, 1:-1, :-2] * self.pre_factors[5][:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 20 == 0:
                self.converged, self.D_eff = self.check_convergence(verbose, conv_crit, start, iter_limit)
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter%2]
            self.conc[:, 1:-1, 1:-1, 1:-1] += out

        if len(np.array([self.VF[z] for z in self.VF.keys() if z!=0]))>0:
            self.D_mean=np.sum(np.array([self.VF[z]*(1/(2*self.cond[z])) for z in self.VF.keys() if z!=0]))
        else:
            self.D_mean=0
        self.tau = self.D_mean/self.D_eff if self.D_eff != 0 else cp.inf
        self.end_simulation(iter_limit, verbose, start)
        return self.tau
    def check_convergence(self, verbose, conv_crit, start, iter_limit):
        # print progress
        if self.iter % 100 == 0:
            loss, flux = self.check_vertical_flux(conv_crit)
            if verbose=='per_iter':
                print(loss)
            if abs(loss) < conv_crit or np.isnan(loss).item():
                self.converged = True
                b, x, y, z = self.cpu_img.shape
                flux *= (x+1)/(y*z)
                try:
                    return True, flux.get()
                except AttributeError as e:
                    return True, flux

        # increase precision to double if currently single
        if self.iter >= iter_limit:
            if self.precision == cp.single:
                print('increasing precision to double')
                self.iter = 0
                self.conc = cp.array(self.conc, dtype=cp.double)
                self.nn = cp.array(self.nn, dtype=cp.double)
                self.precision = cp.double
            else:
                print('Did not converge in the iteration limit')
                try:
                    return True, flux.get()
                except AttributeError as e:
                    return True, flux
        return False, False

    def check_vertical_flux(self, conv_crit):
        vert_flux = (self.conc[:, 1:-1, 1:-1, 1:-1] - self.conc[:, :-2, 1:-1, 1:-1]) * self.pre_factors[1][:, :-2, 1:-1, 1:-1]
        vert_flux[self.nn == cp.inf] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if abs(fl).min()==0:
            return 0, cp.array([0], dtype=self.precision)
        return err, fl.mean()


class eRDMSolver(Solver):
    '''Nguyen et al. 2020
    cupy doesn't support complexe number yet'''
    def __init__(self, img, precision=np.double, bc=(-0.5, 0.5), freq=1, c_dl=0.01):
        self.omega = 2 * np.pi * freq
        self.c_dl = c_dl
        assert precision is np.double, 'Complexe number needs double precision'
        bc = (bc[0]+1j*0, bc[1]+1j*0)
        super().__init__(img, precision, bc)
        print(f'The volume fraction of the computed phase is {self.VF}')

    def init_conc(self, img):
        bs, x, y, z = img.shape
        sh = 1 / (x * 2)
        vec = cp.linspace(self.top_bc + sh, self.bot_bc - sh, x)
        for i in range(2):
            vec = cp.expand_dims(vec, -1)
        vec = cp.expand_dims(vec, 0)
        vec = vec.repeat(z, -1)
        vec = vec.repeat(y, -2)
        vec = vec.repeat(bs, 0)
        vec = vec + 1j * 0 * vec

        return self.pad(img * vec, [self.top_bc * 2, self.bot_bc * 2]) #todo: init for the phase == 1

    
    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2):
        """
        run a solve simulation

        :param iter_limit: max iterations before aborting, will attempt double for the same no. iterations
        if initialised as singles
        :param verbose: Whether to print tau. Can be set to 'per_iter' for more feedback
        :param conv_crit: convergence criteria, minimum percent difference between
        max and min flux through a given layer
        :return: tau
        """
        start = timer()
        while not self.converged:
            # note: out is the average concentration matrix of its surroundings (excluded solid voxels)
            out = self.conc[:, 2:, 1:-1, 1:-1] + \
                  self.conc[:, :-2, 1:-1, 1:-1] + \
                  self.conc[:, 1:-1, 2:, 1:-1] + \
                  self.conc[:, 1:-1, :-2, 1:-1] + \
                  self.conc[:, 1:-1, 1:-1, 2:] + \
                  self.conc[:, 1:-1, 1:-1, :-2]
            out /= self.nn
            if self.iter % 20 == 0:
                lt = abs(cp.sum(out[:, 0]) - self.ph_top)
                lb = abs(cp.sum(out[:, -1]) - self.ph_bot)
                self.converged, D_rel = self.check_convergence(lt, lb, verbose, conv_crit, start, iter_limit)
                if verbose:
                    print(f'iter {self.iter}: D_rel = {D_rel}')
            out -= self.crop(self.conc, 1)
            out *= self.cb[self.iter%2] # note: interesting!
            self.conc[:, 1:-1, 1:-1, 1:-1] += out
            self.iter += 1
        self.D_mean = self.D_0
        self.tau=self.VF/D_rel if D_rel != 0 else cp.inf
        self.D_eff=self.D_mean*D_rel
        self.end_simulation(iter_limit, verbose, start)
        return self.tau


class eSCMSolver(Solver):
    """Nguyen et al. 2020"""
    def __init__(self, img, sep_img=None, precision=cp.double, bc=(0, 1), D_0=1, freq=1, c_dl=0.01, kappa_0=4.6e-2, dx=20e-9):
        # clear gpu memory
        self.clean_gpu()

        # init model (override the whole init method) #todo: trim down later
        # init params
        self.inf = 1e40 #cp.inf will throw NaN in complex array manipulation, see below
        self.precision = precision
        self.assign_omega(freq)
        self.c_dl = c_dl # unit: F.m^-2
        assert precision is cp.double, 'Prefer needs double precision'
        bc = (bc[0]+0j, bc[1]+0j)
        self.kappa_eff= kappa_0 # unit: S.m^-1
        self.top_bc, self.bot_bc = bc # todo: assume infinite conductivity, should solve phi1 separately using multiphase
        self.ph_top = cp.sum(img[:, 0]) * self.top_bc
        self.ph_bot = cp.sum(img[:, -1]) * self.bot_bc
        self.dx = dx # unit: m
        self.dA = dx**2 # unit: m^2
        self.D_0 = D_0  # unit:m^2.s^-1
        self.VF = cp.mean(img)

        # init mats
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        # gen separator
        if sep_img is None:
            #convention: [throughplan_diff_direction, in-plan1, in-plan2]
            sep_img = np.random.rand(img.shape[0], 25, img.shape[2], img.shape[3])
            sep_img[sep_img > 0.1] = 1 # 80% of porosity #todo: Celgard 2500 of 25um, should be tunable or data from V.Wood's group
            sep_img[sep_img != 1] = 0 # solid
        if len(sep_img.shape) == 3:
            sep_img = np.expand_dims(sep_img, axis=0)
        self.cpu_img = img
        self.cpu_sep_img = sep_img
        img = cp.array(img)
        sep_img = cp.array(sep_img)
        
        self.w = 2 - cp.pi / (1.5 * (2 * self.cpu_img.shape[1] + self.cpu_sep_img.shape[1]))
        self.phi2 = self.init_phi2(img, sep_img)
        self.phi1 = self.init_phi1(img, sep_img) 
        self.nn = self.init_nn(cp.concatenate([img, sep_img, cp.flip(img, axis=(1, 2, 3))], axis=1))
        self.NN_tot = self.init_NN_tot(img, sep_img)
        self.cb = self.init_cb(cp.concatenate([img, sep_img, cp.flip(img, axis=(1, 2, 3))], axis=1))
        self.prefactor = self.NN_tot - self.nn
        self.prefactor[cp.where(cp.isinf(self.prefactor) == True)] = 0
        self.prefactor[cp.where(cp.isnan(self.prefactor) == True)] = 0
        self.prefactor[cp.where(self.prefactor < -100)] = 0 #todo: fixme: note: why?
        self.print_state(self.prefactor, 'pref')
        self.print_state(self.nn, 'nn')
        self.print_state(self.NN_tot, 'NN_tot')
        self.print_state(self.phi1, 'phi1')
        self.print_state(self.phi2, 'phi2')

        # solver
        bs, x, y, z = self.cpu_img.shape
        self.L_A = x / (z * y)
        self.converged = False
        self.semi_converged = False
        self.iter=0
        img = None

        # Results
        self.tau=None
        self.D_eff=None
        self.D_mean=None
        self.Z = None
    
    def assign_omega(self, freq):
        self.omega = 2 * cp.pi * freq # unit: rad

    def clean_gpu(self):
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except AttributeError as e: #for no gpu case
            print(e) #todo: prefer logging module
            pass # do nothing
    
    def pad(self, img, vals=[0] * 6):
        while len(vals) < 6:
            vals.append(0)
        to_pad = [1]*8
        to_pad[:2] = (0, 0)
        img = cp.pad(img, np.reshape(to_pad, (4, 2)), 'constant')
        img[:, 0], img[:, -1] = vals[:2]
        img[:, :, 0], img[:, :, -1] = vals[2:4]
        img[:, :, :, 0], img[:, :, :, -1] = vals[4:]
        return img

    def init_phi2(self, img, sep_img):
        # init phi2 
        bs, x, y, z = img.shape
        sh = 1 / (x * 2) #todo: what's this??
        # flip tb & flip lr & flip uf
        template = cp.concatenate([img, sep_img, cp.flip(img, axis=(1, 2, 3))], axis=1)

        phi2= cp.linspace(cp.real(self.top_bc) + sh, cp.real(self.bot_bc) - sh, template.shape[1])
        for i in range(2):
            phi2 = cp.expand_dims(phi2, -1)
        phi2 = cp.expand_dims(phi2, 0)
        phi2 = phi2.repeat(z, -1)
        phi2 = phi2.repeat(y, -2)
        phi2 = phi2.repeat(bs, 0)
        self.print_NaN(phi2, '_phi2')
        phi2 = phi2 + 0j
        self.print_NaN(phi2, 'phi2_complexe')
        phi2 = template * phi2
        self.print_NaN(phi2, 'phi2_final')

        return self.pad(phi2, [self.top_bc, self.bot_bc]) 

    def init_phi1(self, img, sep_img):
        top = cp.zeros_like(img)
        bot = cp.ones_like(img)
        phi1 = cp.concatenate([top, cp.zeros_like(sep_img), bot],axis=1)
        return self.pad(phi1, [0, 1])

    def init_NN_tot(self, img, sep_img):
        if self.nn is None:
            raise ValueError('Please init the Mat Neareat Neighbour self.nn first.')
        NN_tot = self.pad(cp.ones(\
            (img.shape[0], 2*img.shape[1]+sep_img.shape[1], img.shape[2], img.shape[3])
            ), [1]*2)
        NN_tot = NN_tot[:, 2:, 1:-1, 1:-1] + \
                NN_tot[:, :-2, 1:-1, 1:-1] + \
                NN_tot[:, 1:-1, 2:, 1:-1] + \
                NN_tot[:, 1:-1, :-2, 1:-1] + \
                NN_tot[:, 1:-1, 1:-1, 2:] + \
                NN_tot[:, 1:-1, 1:-1, :-2]
        # let NN_tot in solid phase equals to 'inf' too
        solid_phase = cp.concatenate([1-img, cp.zeros_like(sep_img), cp.flip(1-img, axis=(1, 2, 3))], axis=1)
        NN_tot[solid_phase == 1] == self.inf

        # override the separator part to make sure prefactor (N_tot - NN) = 0 in separator
        NN_tot[:, img.shape[1]:-img.shape[1], ] = \
            self.nn[:, img.shape[1]:-img.shape[1], ]
        return NN_tot

    def init_nn(self, img):
        img2 = self.pad(self.pad(img, [1, 1])) #todo: [2, 2]?
        nn = cp.zeros_like(img2, dtype=self.precision)
        # iterate through shifts in the spatial dimensions
        for dim in range(1, 4):
            for dr in [1, -1]:
                nn += cp.roll(img2, dr, dim) #todo: more degree of liberty, higher is the value in nn
        # remove the two paddings
        nn = self.crop(nn, 2)
        nn[img <= 0.2] = self.inf #note: should use cp.inf or create all nan nn in cupy
        self.print_NaN(nn, 'nn')
        nn[nn <= 0.2] = self.inf
        self.print_NaN(nn, 'nn')

        return nn
    
    def solve(self, iter_limit=5000, verbose=True, conv_crit=2*10**-2, plot=True):
        start = timer()
        constant = self.c_dl * 1j * self.omega * self.dA / self.kappa_eff
        print(f'constant: {constant}')
        # self.prefactor = self.NN_tot - self.nn
        # self.prefactor = cp.array(self.prefactor)
        # self.prefactor[cp.where(cp.isinf(self.prefactor) == True)] = 0
        # self.prefactor[cp.where(cp.isnan(self.prefactor) == True)] = 0
        #note: the following has no effect? self.prefactor[(self.prefactor==cp.inf)|(self.prefactor==-cp.inf)|(self.prefactor==cp.nan)] = 0
        
        # allocate variables to GPU at the last moment, to save GPU resouces
        # prefactor, phi1_crop, phi2, nn, cb
        self.prefactor = cp.array(self.prefactor) + 0j
        # self.nn = cp.array(self.nn)
        # self.phi2 = cp.array(self.phi2)
        phi1_crop = cp.array(self.phi1[:, 1:-1, 1:-1, 1:-1]) + 0j
        # self.cb = cp.array(self.cb)

        # loop it
        while not self.converged: 
            # self.print_NaN(constant, 'constant')
            # self.print_NaN(self.prefactor, 'prefactor')
            # self.print_NaN(self.phi2, 'phi2')
            # self.print_NaN(phi1_crop, 'phi1')
            out = self.phi2[:, 2:, 1:-1, 1:-1] + \
                    self.phi2[:, :-2, 1:-1, 1:-1] + \
                    self.phi2[:, 1:-1, 2:, 1:-1] + \
                    self.phi2[:, 1:-1, :-2, 1:-1] + \
                    self.phi2[:, 1:-1, 1:-1, 2:] + \
                    self.phi2[:, 1:-1, 1:-1, :-2] + \
                    self.prefactor * constant * phi1_crop
            if verbose:
                self.print_NaN(out, 'out')
            out /= (self.nn + self.prefactor * constant) # fixme: here throws mysterious NaN in cupy but not in numpy
            if verbose:
                self.print_NaN(out, 'out')

            if self.iter % 100 == 0:
                self.converged, Z_rel = self.check_convergence(verbose, conv_crit, start, iter_limit)
                if verbose:
                    print(f'iter: {self.iter}, Z_rel: {Z_rel}')
            if self.iter % 1000 == 0:
                if plot:
                    self.plot_phis()
                    self.plot_pre_Ntot_nn()

            out -= self.crop(self.phi2, 1)
            out *= self.cb[self.iter%2] # todo: might be different for complexe?
            self.phi2[:, 1:-1, 1:-1, 1:-1] += out
            self.iter += 1

        # method1
        # self.Z = cp.mean(self.phi2[:, 1:-1, 1:-1, 1:-1] \
        #      / self.kappa_eff * cp.diff(self.phi2, axis=1)[:, 1:, 1:-1, 1:-1]) #todo: cp.diff axis=1 on the diffusion direction?
        # method2
        # self.Z_top = cp.mean(out[:, 0]) - self.top_bc
        # self.Z_bot = cp.mean(out[:, -1]) - self.bot_bc
        # method3
        self.I_sep = self.get_separator_current()
        self.Z = (self.bot_bc - self.top_bc) / self.I_sep
        
        # print(f'Z_top: {self.Z_top}, Z_bot: {self.Z_bot}')
        print(f'\nZ: {self.Z}')

        self.end_simulation(iter_limit, verbose, start)
        return self.tau
    
    def solve4freqs(self, freq_range=[0, 5, 5], iter_limit=5000, verbose=True, conv_crit=5e-2, plot=True):
        assert len(freq_range) == 3, 'Only give a low bound and a high bound frequencies'
        self.Zs = []
        for freq in tqdm(cp.logspace(cp.log(freq_range[0]), cp.log(freq_range[1]), freq_range[2])):
            self.clean_gpu() # don't clean phi1, NN_tot, etc
            # only need to re init the phi2 variable at each iter
            self.__init__(self.cpu_img, self.cpu_sep_img, freq=freq)
            s = self.solve(iter_limit=iter_limit, verbose=verbose, conv_crit=conv_crit, plot=plot)
            self.Zs.append(self.Z) 
            print(f'Z: {self.Z}')
            if plot:
                self.plot_phis()
                self.plot_pre_Ntot_nn()

        self.Zs = cp.asarray(self.Zs)
        # resolution = 20e-9  # m/voxel
        # normFactor = self.kappa_eff / resolution #(L_x / vox_x)
        # print(f'normalization factor: {normFactor}')
        cp.savetxt('./data.txt', self.Zs)

        plt.figure()
        plt.title('Nyquist Plot')
        print(self.Zs)
        # plt.plot(cp.real(self.Zs) * normFactor, -cp.imag(self.Zs) * normFactor, 'x-')
        plt.plot(cp.real(self.Zs), -cp.imag(self.Zs), 'x-')
        plt.show()

    def print_NaN(self, array, arr_name=''):
        try:
            percentage = len(cp.argwhere(cp.isnan(array) == True))/array.size *100
            if percentage >= 0.01:
                print(f'found {percentage} % NaN voxel in {arr_name}')
        except Exception as e:
            print('****************')
            print(e)
    
    def print_state(self, array, arr_name=''):
        try:
            print(f'arr: {arr_name}, min:{array.min()}, max:{array.max()}')
        except Exception as e:
            print(e)

    def plot_pre_Ntot_nn(self):
        plt.figure()
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        try:
            im1 = ax1.imshow(np.real(self.prefactor[0, :, self.prefactor.shape[2]//2, :]))
        except TypeError as e:
            im1 = ax1.imshow(cp.real(self.prefactor[0, :, self.prefactor.shape[2]//2, :]).get()) 
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('prefactor')
        self.print_NaN(self.prefactor, 'prefactor')
        try:
            im2 = ax2.imshow(np.real(self.nn[0, :, self.nn.shape[2]//2, :]))
        except TypeError as e:
            im2 = ax2.imshow(cp.real(self.nn[0, :, self.nn.shape[2]//2, :]).get())
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('nn')
        self.print_NaN(self.nn, 'nn')
        try:
            im3 = ax3.imshow(np.real(self.NN_tot[0, :, self.NN_tot.shape[2]//2, :]))
        except TypeError as e:
            im3 = ax3.imshow(cp.real(self.NN_tot[0, :, self.NN_tot.shape[2]//2, :]).get())
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('NN_tot')
        self.print_NaN(self.NN_tot, 'NN_tot')
        plt.tight_layout()
        plt.show()

    def plot_phis(self):
        plt.figure(figsize=(8, 16))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        try:
            im = ax1.imshow(cp.real(self.phi2[0, :, self.phi2.shape[2]//2, :])) 
        except TypeError as e:
            im = ax1.imshow(cp.real(self.phi2[0, :, self.phi2.shape[2]//2, :]).get())
        ax1.set_title('phi2 real')
        fig.colorbar(im, ax=ax1)
        try:
            im = ax2.imshow(cp.imag(self.phi2[0, :, self.phi2.shape[2]//2, :]))
        except TypeError as e:
            im = ax2.imshow(cp.imag(self.phi2[0, :, self.phi2.shape[2]//2, :]).get())
        ax2.set_title('phi2 imag')
        fig.colorbar(im, ax=ax2)
        try:
            im = ax3.imshow(cp.real(self.phi1[0, :, self.phi1.shape[2]//2, :]))
        except TypeError as e:
            im = ax3.imshow(cp.real(self.phi1[0, :, self.phi1.shape[2]//2, :]).get())
        ax3.set_title('phi1 real')
        fig.colorbar(im, ax=ax3)
        try:
            im = ax4.imshow(cp.imag(self.phi1[0, :, self.phi1.shape[2]//2, :]))
        except TypeError as e:
            im = ax4.imshow(cp.imag(self.phi1[0, :, self.phi1.shape[2]//2, :]).get())
        ax4.set_title('phi1 imag')
        fig.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.show()
    
    def check_vertical_flux(self, conv_crit):
        vert_flux = self.phi2[:, 1:-1, 1:-1, 1:-1] - self.phi2[:, :-2, 1:-1, 1:-1]
        # vert_flux[self.phi2[:, :-2, 1:-1, 1:-1] == 0] = 0
        # vert_flux[self.phi2[:, 1:-1, 1:-1, 1:-1] == 0] = 0
        fl = cp.sum(vert_flux, (0, 2, 3))[1:-1]
        err = (fl.max() - fl.min())*2/(fl.max() + fl.min())
        if err < conv_crit or np.isnan(err).item():
            return True
        if fl.min() == 0:
            return 'zero_flux'
        return False
    
    def get_separator_current(self):
        # curr = cp.diff(self.kappa_eff * self.phi2[:, 1:-1, 1:-1, 1:-1], axis=1) #unit A.m^-1 only verticle current are considered
        # curr = curr[:, self.cpu_img.shape[1]:-self.cpu_img.shape[1], :, :] #take the separator part
        # I_sep = cp.mean(curr[:, -1, :, :]) / self.dA # unit: A.m^-3

        # assume conserved current on the verticle direction in the middle of the separator
        num_liq_vox = self.cpu_sep_img[self.cpu_sep_img.shape[0] // 2].size
        print(num_liq_vox)
        I_sep = cp.sum(self.kappa_eff * self.dx * \
            (self.phi2[:, self.phi2.shape[1] // 2, 1:-1, 1:-1] - self.phi2[:, self.phi2.shape[1] // 2 - 1, 1:-1, 1:-1]))\
                /(num_liq_vox * self.dA)
        return I_sep

    def check_convergence(self, verbose, conv_crit, start, iter_limit):
        Z =  (self.top_bc - self.bot_bc) / self.get_separator_current()
        if self.iter <= iter_limit:
            return False, Z
        else:
            self.plot_phis()
            self.plot_pre_Ntot_nn()
            return True, Z