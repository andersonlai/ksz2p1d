from numba import njit, prange
import healpy as hp
import numpy as np
from scipy.special import loggamma
from scipy.interpolate import interp1d
import camb
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

@njit
def w3j(j1,j2,j3):
    if not ((j1+j2+j3) % 2 == 0) or j3 > j1+j2 or j3 < np.abs(j2-j1):
        return 0
    else:
        g = (j1+j2+j3)/2
    phase = np.power(-1,g)
    return phase*np.exp( 0.5*(loggamma(2*g-2*j1+1)+loggamma(2*g-2*j2+1)+loggamma(2*g-2*j3+1) \
            -loggamma(2*g+2))+loggamma(g+1)-(loggamma(g-j1+1)+loggamma(g-j2+1)+loggamma(g-j3+1)) )

@njit(parallel=True)
def Nvv_inv(nvv,l_, c_tilde_T,c_tilde_g,cl_tau_g,j1_max,j2_max,lcutoff):
    for k in range(len(nvv)):
        nvv_l = 0
        for j1 in prange(0,j1_max):
            if (j1 > lcutoff):
                for j2 in prange(0,j2_max):
                    if (j2 > lcutoff):
                        nvv_l += np.power(c_tilde_T[j1],-1)*np.power(cl_tau_g[j2],2)*np.power(c_tilde_g[j2],-1)\
                        *(2*j1+1)*(2*j2+1)*(2*l_[k]+1)/(4*np.pi)*w3j(j1,j2,l_[k])**2

        nvv[k] = nvv_l        
    return nvv/(2*l_+1)

@njit(parallel=True)
def Nvv_inv_cross(nvv,l_, c_tilde_T1, c_tilde_T2, c_tilde_T12, c_tilde_g,cl_tau_g,j1_max,j2_max, lcutoff):
    for k in range(len(nvv)):
        nvv_l = 0
        for j1 in prange(0,j1_max):
            if (j1 > lcutoff):
                for j2 in prange(0,j2_max):
                    if (j2 > lcutoff):
                        nvv_l += np.power(c_tilde_T1[j1],-1)*np.power(c_tilde_T2[j1],-1)\
                        *c_tilde_T12[j1]*np.power(cl_tau_g[j2],2)*np.power(c_tilde_g[j2],-1)\
                        *(2*j1+1)*(2*j2+1)*(2*l_[k]+1)/(4*np.pi)*w3j(j1,j2,l_[k])**2

        nvv[k] = nvv_l        
    return nvv/(2*l_+1)

def rec_vr(haloalms, combalm, C_dd, C_taud, C_TT, nside, nside_out, lcutoff):
    alm_ = hp.almxfl(combalm, np.power(C_TT, -1), inplace=False)
    A_T = hp.alm2map(alm_, nside)
    
    haloalms_ = hp.almxfl(haloalms,(np.divide(C_taud, C_dd, out=np.zeros_like(C_taud), where=C_dd!=0)))
    B_g = hp.alm2map(haloalms_, nside=nside)
    
    l_ = np.arange(0,3*nside_out+5,3)
    l_max = C_dd.shape[0]-1
    #l_max = l_arr.shape[0]-1
    nvv_i = Nvv_inv(np.zeros_like(l_),l_, C_TT, C_dd, C_taud,l_max,l_max, lcutoff)
    #nvv_i = Nvv_inv(l_, C_TT, C_dd, C_taud,l_max)
    rec_noise_inv_interp = interp1d(l_ ,nvv_i, bounds_error=False)
    
    rec_noise = np.power(rec_noise_inv_interp(np.arange(3*nside_out+1)),-1)
    vrec_lm_bare = hp.map2alm(A_T*B_g,lmax=3*nside_out)
    
    return vrec_lm_bare, rec_noise

def ud_grade(map, nside_out):
    alm = hp.map2alm(map, lmax=3*nside_out-1)
    map = hp.alm2map(alm, nside=nside_out)

    return map

class estimator:
    def initialize(self, lmax, nside):
        self.lmax = lmax
        self.nside = nside
        self.Tcmb = 2.726

    def load_masks(self, Tmask_fname, gmask_fname, overlapmask_in_fname, overlapmask_out_fname):
        self.Tmask = hp.read_map(Tmask_fname)
        self.gmask = hp.read_map(gmask_fname)
        self.overlapmask_in = hp.read_map(overlapmask_in_fname)
        self.overlapmask_out = hp.read_map(overlapmask_out_fname)

        self.fsky_T = np.mean(self.Tmask**2)
        self.fsky_g = np.mean(self.gmask**2)
        self.fsky_overlap = np.mean(self.overlapmask_in**2)

    def load_data(self, Tmap_dir, odmap_dir):
        # load temperature maps
        freqs = []
        CTTs = []
        Talms = []
        
        for i in tqdm(os.listdir(Tmap_dir)):
            if 'Tmap' in i:
                Tmap = hp.read_map(Tmap_dir+i+'/Tmap_nside'+str(self.nside)+'.fits')
                CTT, Talm = hp.anafast(Tmap*self.overlapmask_in, lmax=self.lmax, alm=True)
                CTTs.append(CTT)
                Talms.append(Talm)
                freqs.append((i.split('_'))[1])

        self.freqs = freqs
        self.CTTs = CTTs
        self.Talms = Talms

        # load overdensity maps in alms and power spectrums
        galms = []
        Cggs = []
        nums = []

        for i in tqdm(os.listdir(odmap_dir)):
            if len(i.split('_')) > 1:
                if (i.split('_'))[1] == 'overdensity':
                    map = hp.read_map(odmap_dir+i)
                    Cgg, galm = hp.anafast(map*self.overlapmask_in, lmax = self.lmax, alm=True)

                    nums.append(int(i.split('_')[3][3:]))
                    Cggs.append(Cgg)
                    galms.append(galm)

        args = np.argsort(nums)
        self.galms = np.array(galms)[args]
        self.Cggs = np.array(Cggs)[args]

    def load_Cgtau(self, Cgtau_dir):
        f = np.load(Cgtau_dir+'fiducials.npz')
        Cmtau2hs = f['Cmtau2hs']
        Cgtau1hs = f['Cgtau1hs']
        bg = f['bg']
        Cgtau2hs = bg[:, None]*Cmtau2hs
        self.Cgtaus = (Cgtau1hs + Cgtau2hs)[:,:self.lmax+1]

    def get_estimator(self, nside_out, lcutoff):
        vmaps_dict = {}
        vmaps_meansub_dict = {}
        Nvvs_dict = {}

        for i in range(len(self.freqs)):
            freq = self.freqs[i]
            CTT = self.CTTs[i]/self.fsky_overlap
            Talm = self.Talms[i]

            vmaps = []
            Nvvs = []

            for j in tqdm(range(self.Cggs.shape[0])):
                galm = self.galms[j]
                Cgtau = self.Cgtaus[j]
                Cgg = self.Cggs[j]/self.fsky_overlap

                valms_rec_bare, recnoise = rec_vr(galm, Talm, Cgg, Cgtau, CTT, self.nside, nside_out, lcutoff)

                valms_rec = hp.almxfl(valms_rec_bare, recnoise)
                vmap_rec = hp.alm2map(valms_rec, nside=nside_out)
                # Cvv_rec = hp.anafast(vmap_rec)

                Nvvs.append(recnoise)
                vmaps.append(vmap_rec)
                # Cvvs_rec.append(Cvv_rec)

            vmaps_meansub = []
            for j in range(self.Cggs.shape[0]):
                vmap_meansub = (vmaps[j] - np.mean(vmaps[j][self.overlapmask_out.astype(int) == 1]))*self.overlapmask_out
                vmaps_meansub.append(vmap_meansub)

            vmaps = np.array(vmaps)
            vmaps_meansub = np.array(vmaps_meansub)
            Nvvs = np.array(Nvvs)
            
            vmaps_dict[freq] = -1*vmaps
            vmaps_meansub_dict[freq] = -1*vmaps_meansub
            Nvvs_dict[freq] = Nvvs

        self.vmaps_dict = vmaps_dict
        self.vmaps_meansub_dict = vmaps_meansub_dict
        self.Nvvs_dict = Nvvs_dict