import numpy as np
import healpy as hp
import hmvec as hm
import camb
from camb import model, initialpower
import cosmo_constants as cc
from scipy.interpolate import RegularGridInterpolator
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pymaster as nmt
import os

def check_fname(fname):
    if os.path.exists(fname) == False:
        print('the file: ', fname, ' does not exist, please check the input')
        return False
    else:
        print('Found the file in: ', fname, ', proceeding...')
        return True

def check_dir(dir):
    if os.path.exists(dir) == False:
        print('directory not found, creating directory...')
        os.mkdir(dir)
        print('directory ', dir, ' is created.')
    else:
        return

def ud_grade(m, nside_out):
    alm = hp.map2alm(m, lmax = 3*nside_out-1)
    return hp.alm2map(alm, nside_out)

# tons of fiducial functions
def project(C, ellgrid, ells):
    interp = interp1d(ellgrid, C)
    return interp(ells)

def ne0z():
    # from Mortiz's code, true for z < 3 (we have z<3 in Yuukis sims)
    G_SI = 6.674e-11   
    mProton_SI = 1.673e-27
    H100_SI = 3.241e-18
    chi = 0.86
    me = 1.14
    gasfrac = 0.9
    ombh2 = 0.022
    omgh2 = gasfrac*ombh2
    ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me                   
    return ne0_SI        # unit in m^-3 

def electronfactor(z):
    thompson_SI = 6.6524e-29         # unit m^2
    meterToMegaparsec = 3.241e-23
    aas = 1./(1.+z)
    factor = thompson_SI*ne0z()*aas**(-2.)/meterToMegaparsec#*2.725*10.**6
    return factor

def initialize_camb(zmin, zmax, ks, zs):
    zs_skip = np.linspace(zmin, zmax+0.02, 255)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*cc.h, ombh2=cc.ombh2, omch2=cc.omch2)
    pars.InitPower.set_params(ns=cc.ns)
    pars.NonLinear = model.NonLinear_both
    pars.set_matter_power(redshifts=zs_skip, kmax=max(ks))
    results = camb.get_results(pars)

    results.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)
    Pk = results.get_matter_power_interpolator(hubble_units=False, k_hunit=False)
    pk_i = Pk.P(zs, ks)

    rs = results.comoving_radial_distance(zs)
    asarr = 1/(1+zs)
    Hs = results.hubble_parameter(zs)
    fs = (results.get_Omega('baryon',zs)+results.get_Omega('cdm',zs))**0.55

    faHs = asarr*fs*Hs
    
    return rs, Hs

class p_of_zl():
    def __init__(self,zs,rs,intrp_obj):
        self.intrp_obj = intrp_obj
        self.xx, self.yy = np.meshgrid(zs, rs,indexing='ij')
    def __call__(self,l):
        return np.fliplr(self.intrp_obj((self.xx, np.log10(l/self.yy)))).diagonal()

def parallel_executor(func,iterables):    
    with mp.Pool() as p:
        result = p.map(func,iterables)
    return result

def Cgg_limber(pk_i, r_arr, H_arr, z_arr, k_arr, l_arr, Wdndz):
    
    log_pk_interpolator = RegularGridInterpolator((z_arr, np.log10(k_arr)), np.log10(pk_i), bounds_error=False, fill_value=None)
    p_zl = np.array(parallel_executor(p_of_zl(z_arr,r_arr,log_pk_interpolator),l_arr))
    C_l_limber = np.trapz(Wdndz**2*H_arr/r_arr**2*10**p_zl, z_arr)
    C_l_limber /= 3e5
    # C_l_limber = np.trapz(Wdndz1*Wdndz2*cc.c/H_arr/r_arr**2*10**p_zl, z_arr)
    
    return C_l_limber

def Cgtau_limber(pk_i, r_arr, H_arr, z_arr, k_arr, l_arr, Wdndz1, Wdndz2):
    
    log_pk_interpolator = RegularGridInterpolator((z_arr, np.log10(k_arr)), np.log10(pk_i), bounds_error=False, fill_value=None)
    p_zl = np.array(parallel_executor(p_of_zl(z_arr,r_arr,log_pk_interpolator),l_arr))
    C_l_limber = np.trapz(Wdndz1*Wdndz2*(1+z_arr)**2/r_arr**2*10**p_zl, z_arr)
    # C_l_limber /= 3e5
    
    return C_l_limber

class fiducial:
    def initialize_profile(self, profile_fname):
        cond = check_fname(profile_fname)
        
        if cond == False:
            return
        
        if cond == True:
            self.profile_fname = profile_fname
            print('loading redshift bin configuration from the following photo-z profile: ', profile_fname)
            f = np.load(profile_fname)
            self.zgrid = f['zgrid']
            self.convolved_normalized_profiles = f['conv_normed_profiles']
            self.profile_edges = f['zbins']
            self.sigmas = f['photo_z_err_bin']
            self.nbins = f['zbins'].shape[0] - 1

    def inititalize_hod_parameters(self, mmin=1e10, mmax=5e15, kmin=-5, kmax=2, kticks=8192*4, lmax=12000, lspace=5):
        self.mmin = mmin
        self.mmax = mmax
        self.kmin = kmin
        self.kmax = kmax
        self.kticks = kticks
        self.ks = np.logspace(kmin, kmax, kticks)
        self.lmax = lmax
        self.ellgrid = np.arange(1, lmax, lspace)

        print('you have definied the following hod parameters: ')
        print('minimum halo mass = ', self.mmin)
        print('maximum halo mass = ', self.mmax)
        print('minimum wavenumber = ', self.ks[0])
        print('maximum wavenumber = ', self.ks[-1])
        print('wavenumber resolution (in logspace) = ', self.ks.shape[0])
        print('maximum multipole = ', self.lmax)

        self.nsigma = 2.5
        self.bin_mid = (self.profile_edges[1:] + self.profile_edges[:-1])/2
        self.zmins = self.bin_mid - self.nsigma*self.sigmas
        self.zmaxs = self.bin_mid + self.nsigma*self.sigmas
        
    def initialize_electron_profile(self, xmax=100, nxs=10000, families='AGN'):
        self.xmax=xmax
        self.nxs=nxs
        self.families=families

        print('you have definied the followin electron profile parameters: ')
        print('xmax = ', xmax)
        print('nxs = ', nxs)
        print('family = ', families)

        self.electron_factor = electronfactor(0)

    def get_fiducials(self, ells = None):
        if ells is not None:
            self.ells = ells
        else:
            self.ells = np.arange(1, self.ellgrid[-1], 1)
            
        Cmtau1hs = []
        Cmtau2hs = []

        Cgtau1hs = []
        Cgtau2hs = []

        Cmm1hs = []
        Cmm2hs = []

        Cgg1hs = []
        Cgg2hs = []

        for i in range(self.nbins):
            print('Computing fiducials for bin', i+1)
            
            zmin = self.zmins[i]
            zmax = self.zmaxs[i]
            ms = np.geomspace(self.mmin, self.mmax, 50)

            zs = np.linspace(zmin, zmax, 100, endpoint=True)
            
            Ws_interp = interp1d(self.zgrid, self.convolved_normalized_profiles[i])
            W = Ws_interp(zs)/np.trapz(Ws_interp(zs), zs)
            We = self.electron_factor

            rs,Hs = initialize_camb(zmin, zmax, self.ks, zs)
            hcos = hm.HaloModel(zs, self.ks, ms)

            print('constructing galaxy hod...')
            hcos.add_hod(name="g",mthresh=10**10.5+zs*0.)
            
            print('constructing electron hod...')
            hcos.add_battaglia_profile("electron",family=self.families,xmax=self.xmax,nxs=self.nxs)

            print('Computing Cggs...')
            Pgg_2h = hcos.get_power_2halo('g')
            Pgg_1h = hcos.get_power_1halo('g')

            Cgg1h_sparse = Cgg_limber(Pgg_1h, rs, Hs, zs, self.ks, self.ellgrid, W)
            Cgg2h_sparse = Cgg_limber(Pgg_2h, rs, Hs, zs, self.ks, self.ellgrid, W)

            Cgg1hs.append(project(Cgg1h_sparse, self.ellgrid, self.ells))
            Cgg2hs.append(project(Cgg2h_sparse, self.ellgrid, self.ells))

            print('Computing Cgtaus...')
            Pge_2h = hcos.get_power_2halo('g', 'electron')
            Pge_1h = hcos.get_power_1halo('g', 'electron')

            Cgtau1h_sparse = Cgtau_limber(Pge_1h, rs, Hs, zs, self.ks, self.ellgrid, W, We)
            Cgtau2h_sparse = Cgtau_limber(Pge_2h, rs, Hs, zs, self.ks, self.ellgrid, W, We)

            Cgtau1hs.append(project(Cgtau1h_sparse, self.ellgrid, self.ells))
            Cgtau2hs.append(project(Cgtau2h_sparse, self.ellgrid, self.ells))

            print('Computing Cmms...')
            Pmm_2h = hcos.get_power_2halo('nfw')
            Pmm_1h = hcos.get_power_1halo('nfw')

            Cmm1h_sparse = Cgg_limber(Pmm_1h, rs, Hs, zs, self.ks, self.ellgrid, W)
            Cmm2h_sparse = Cgg_limber(Pmm_2h, rs, Hs, zs, self.ks, self.ellgrid, W)

            Cmm1hs.append(project(Cmm1h_sparse, self.ellgrid, self.ells))
            Cmm2hs.append(project(Cmm2h_sparse, self.ellgrid, self.ells))

            print('Computing Cmtaus...')
            Pme_2h = hcos.get_power_2halo('nfw', 'electron')
            Pme_1h = hcos.get_power_1halo('nfw', 'electron')

            Cmtau1h_sparse = Cgtau_limber(Pme_1h, rs, Hs, zs, self.ks, self.ellgrid, W, We)
            Cmtau2h_sparse = Cgtau_limber(Pme_2h, rs, Hs, zs, self.ks, self.ellgrid, W, We)

            Cmtau1hs.append(project(Cmtau1h_sparse, self.ellgrid, self.ells))
            Cmtau2hs.append(project(Cmtau2h_sparse, self.ellgrid, self.ells))

        self.Cmtau1hs = np.array(Cmtau1hs)
        self.Cmtau2hs = np.array(Cmtau2hs)

        self.Cgtau1hs = np.array(Cgtau1hs)
        self.Cgtau2hs = np.array(Cgtau2hs)

        self.Cmm1hs = np.array(Cmm1hs)
        self.Cmm2hs = np.array(Cmm2hs)

        self.Cgg1hs = np.array(Cgg1hs)
        self.Cgg2hs = np.array(Cgg2hs)
        

        print('All redshift bins are completed.')

    def get_bias(self, calibrated_map_dir, mask_fname, bg_lmin, bg_lmax):
        
        self.bg_lmin = bg_lmin
        self.bg_lmax = bg_lmax
    
        map_dirs = os.listdir(calibrated_map_dir)
        calibrated_map_dirs = []
        
        for dir in map_dirs:
            if 'calibrated' in dir:
                calibrated_map_dirs.append(dir)
    
        print('The following list of directories will be used: ', calibrated_map_dirs)
        print('Constructing pesudo Cl for each calbrated overdensity map...')
    
        mask = hp.read_map(mask_fname)
        nside = hp.npix2nside(mask.shape[0])
    
        lmax = 3*nside-1
        ells = np.arange(lmax+1)
    
        print('Apodizing mask...')
        mask_apo = nmt.mask_apodization(mask, 0.3, apotype='C2')
    
        print('Reading calibrated maps...')
        calibrated_odmaps = []
        for dir in calibrated_map_dirs:
            calibrated_odmaps.append(hp.read_map(calibrated_map_dir+dir))
    
        calibrated_odmaps = np.array(calibrated_odmaps)
    
        print('Constructing bandpowers...')
        b = nmt.NmtBin.from_nside_linear(nside, 4)
        leffs = b.get_effective_ells()
    
        fmask = nmt.NmtField(mask_apo, None, spin = 0,n_iter=0)
        wmask = nmt.NmtWorkspace.from_fields(fmask,fmask,b)
    
        Bbl = wmask.get_bandpower_windows().squeeze()
    
        print('Calculating pCl...')
        fs = [nmt.NmtField(mask_apo, [m], n_iter = 0) for m in calibrated_odmaps]
        pcl = np.array([wmask.decouple_cell(nmt.compute_coupled_cell(f,f)) for f in fs]).squeeze()
    
        Cmm = self.Cmm2hs[:, :lmax+1]
        Cmm_Bbl = Cmm@Bbl.T
    
        print('Bbl shape', Bbl.shape)
        print('Cmm shape', Cmm.shape)
    
        leffs_mask = np.ones_like(leffs)
        leffs_mask[leffs < bg_lmin] = 0
        leffs_mask[leffs > bg_lmax] = 0
        leffs_mask = leffs_mask.astype(int)
    
        Cmm_Bbl_mask = Cmm_Bbl[:, leffs_mask]
        pcl_mask = pcl[:, leffs_mask]
    
        bg2 = np.sum(pcl_mask*Cmm_Bbl_mask, axis = 1)/np.sum(Cmm_Bbl_mask*Cmm_Bbl_mask, axis = 1)
    
        self.bg2 = bg2

    def output(self, dir=None):
        mmin_tag = str(self.mmin/(10**np.log10(self.mmin).astype(int)))+ 'e'+str(np.log10(self.mmin).astype(int))
        mmax_tag = str(self.mmax/(10**np.log10(self.mmax).astype(int)))+ 'e'+str(np.log10(self.mmax).astype(int))
        mtag = 'mmin'+mmin_tag+'_mmax'+mmax_tag
        
        kmin_tag = str(self.kmin)
        kmax_tag = str(self.kmax)
        kticks_tag = str(self.kticks)
        ktag = 'kmin'+kmin_tag+'_kmax'+kmax_tag

        etag1 = str(self.xmax)
        etag2 = str(self.nxs)
        etag3 = str(self.families)
        etag = 'xmax'+etag1+'_nxs'+etag2+'_family'+etag3

        ztag = 'sigma'+str(self.nsigma)

        dir_tag = mtag+'_'+ktag+'_'+etag+'_'+ztag+'/'

        if dir is not None:
            dir_full = dir+dir_tag
        else:
            dir_full = dir_tag

        check_dir(dir_full)

        print('now outputing fiducials...')
        np.savez('fiducials.npz', ells = self.ells, Cmm1hs=self.Cmm1hs, Cmm2hs=self.Cmm2hs, Cgg1hs=self.Cgg1hs, Cgg2hs=self.Cgg2hs, Cmtau1hs=self.Cmtau1hs, Cmtau2hs=self.Cmtau2hs, Cgtau1hs=self.Cgtau1hs, Cgtau2hs=self.Cgtau2hs, bg2 = self.bg2)

        print('output completed.')