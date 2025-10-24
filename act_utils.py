import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pixell import enmap, reproject
from astropy.io import fits
from astropy.table import Table
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

def find_freq(fname):
    for t in fname.split("_"):
        if t[0] == 'f':
            return t

def dgrade(map, nside_out):
    alm = hp.map2alm(map, lmax = 3*nside_out - 1)
    map_dg = hp.alm2map(alm, nside = nside_out)
    return map_dg

class act_utils:
    def load_map(self, act_fname, nside=4096):
        self.Tcmb = 2.726
        cond = check_fname(act_fname)
        
        if cond == False:
            return
        
        if cond == True:
            print('loading temperature map from the file name: ', act_fname)
            
            self.Tmap_fname = act_fname
            self.freq = find_freq(self.Tmap_fname)
            self.nside = nside
            
            Tmap = hp.read_map(self.Tmap_fname)
            self.Tmap = dgrade(Tmap, nside)/1e6/self.Tcmb
            # self.Tmap_overlay = np.copy(self.Tmap)

            self.Tmap_tag = 'Tmap_'+self.freq

            self.skymask = None
            self.noisemask = None
            self.clustermask = None
            
            self.skymask_tag = None
            self.noisemask_tag = None
            self.clustermask_tag = None
    
    def load_beam(self, beam_fname):
        cond = check_fname(beam_fname)
        
        if cond == False:
            return
        
        if cond == True:
            print('loading beam file from the file name: ', beam_fname)

            Bell = np.loadtxt(beam_fname)
            
            Bell_interp =  interp1d((Bell.T)[0], (Bell.T)[1])
            lmax_beam = int((Bell.T)[0][-1])

            if lmax_beam <= (3*self.nside-1):
                Bell_ = np.zeros(3*self.nside)
                Bell_[:lmax_beam] = Bell_interp(np.arange(lmax_beam))
            else:
                Bell_ = Bell_interp(np.arange(3*self.nside))
            
            pixwin = hp.pixwin(nside=self.nside, lmax= 3*self.nside - 1)
            self.Bell = Bell_/Bell_[0]*pixwin
            self.Bell_tag = None

    def load_ivarmap(self, ivar_fname):
        cond = check_fname(ivar_fname)
        
        if cond == False:
            return
        
        if cond == True:
            print('loading inverse variance map from the file name: ', ivar_fname)
            ivar_map_hpix = dgrade(hp.read_map(ivar_fname), self.nside)
            Omega_pix_armin2 = 4*np.pi/12/self.nside**2*(180*60/np.pi)**2

            print('converting to noise map...')
            self.noise_map = np.sqrt(np.divide(1,ivar_map_hpix, where = ivar_map_hpix > 0))*np.sqrt(Omega_pix_armin2)

            print('completed.')
        
    def load_skymask(self, mask_fname):
        cond = check_fname(mask_fname)
        
        if cond == False:
            return
        
        if cond == True:
            skymask = hp.read_map(mask_fname)
            print('checking if the nside matches between temperature map and mask...')
            
            if hp.npix2nside(skymask.shape[0]) != self.nside:
                print('does not match, please check.')
                return
            else:
                print('mask loaded.')
                self.skymask = skymask
                self.skymask_fsky = np.mean(skymask**2)
                self.skymask_tag = 'skymasked'
    
    def load_clustermask(self, mask_fname):
        cond = check_fname(mask_fname)
        
        if cond == False:
            return
        
        if cond == True:
            clustermask = hp.read_map(mask_fname)
            print('checking if the nside matches between temperature map and mask...')
            
            if hp.npix2nside(clustermask.shape[0]) != self.nside:
                print('does not match, please check.')
                return
            else:
                print('mask loaded.')
                self.clustermask = clustermask
                self.clustermask_fsky = np.mean(clustermask**2)
                self.clustermask_tag = 'clustermasked'

    def get_noise_mask(self, threshold):
        noisemask = np.ones_like(self.Tmap)
        self.noise_cut = threshold
        noisemask[self.noise_map > threshold] = 0
        self.noisemask = noisemask
        self.noisemask_tag = 'noisemasked'+str(np.round(self.noise_cut,2))

    def beam_deconvolution(self, overwrite = False):
        # check if the map has been deconvolved: "
        if (overwrite == False):
            if self.Bell_tag is not None:
                print('the map has been deconvolved, if you want to redo the deconvolution please set overwrite = True.')
                return
        print('decovolving the masked cmb map...')
        alm = hp.map2alm(self.Tmap_integrated, lmax=3*self.nside-1)
        alm_fl = hp.almxfl(alm, np.divide(1, self.Bell, where = self.Bell != 0))
        self.Tmap_integrated = hp.alm2map(alm_fl, nside = self.nside)
        self.Bell_tag = 'beam_deconvolved'
        print('deconvolution completed.')

    def apply_mask(self):
        print('masks that has been configured')
        if self.skymask_tag is not None:
            print('sky mask: ', self.skymask_tag)
        if self.noisemask_tag is not None:
            print('noise mask: ', self.noisemask_tag)
        if self.clustermask_tag is not None:
            print('tsz cluster mask: ', self.clustermask_tag)
        print('note that recalling this function will overwrite the masks that has been applied to the map')

        integratedmask = np.ones_like(self.Tmap)
        self.Tmap_integrated = np.copy(self.Tmap)
        
        if self.skymask is not None:
            self.Tmap_integrated *= self.skymask
            integratedmask *= self.skymask
        if self.noisemask is not None:
            self.Tmap_integrated *= self.noisemask
            integratedmask *= self.noisemask
        if self.clustermask is not None:
            self.Tmap_integrated *= self.clustermask
            integratedmask *= self.clustermask

        self.integratedmask = integratedmask
        self.fsky_integrated = np.mean(self.integratedmask**2)

        print('The masked CMB map and the integated mask has been generated.')

    # def apply_weighting(self, type='uniform'):
        #try to support fkp weight to the tempearture map#

    def output(self, dir=None):
        dir_tag = self.Tmap_tag
        
        if self.skymask is not None:
            dir_tag += ('_'+self.skymask_tag)
            
        if self.noisemask is not None:
            dir_tag += ('_'+self.noisemask_tag)

        if self.clustermask is not None:
            dir_tag += ('_'+self.clustermask_tag)
        
        if dir is not None:
            dir_full = dir + dir_tag + '/'

        check_dir(dir_full)

        print('writing maps to the following directory: ', dir_full)

        hp.write_map(dir_full+'Tmap_nside'+str(self.nside)+'.fits', self.Tmap_integrated)
        hp.write_map(dir_full+'mask_nside'+str(self.nside)+'.fits', self.integratedmask)

        print('output completed.')