import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyccl as ccl
import scipy as sp
import pymaster as nmt
from tqdm import tqdm

class get_biases():
    def init(self,input_dir, sigma_z_cut, nbins = 20, nside = 256, zs = np.linspace(0.2, 1.25, 500), Omega_c=0.268, Omega_b=0.048, h=0.675, A_s=2.1e-9, n_s=0.96,
             mask_path = '/scratch2/acmlai/ACTDR6/masks/desi_mask_nside256.fits',
             lcut = 6, delta_ell = 6, l_limber = 120, leffs_up=200, leffs_low=50):
        self.nside = nside
        self.nbins = nbins
        self.lmax = 3*nside-1
        self.ells = np.arange(self.lmax+1)
        self.zs = zs
        self.Omega_c = Omega_c
        self.Omega_b = Omega_b
        self.h = h
        self.A_s = A_s
        self.n_s = n_s
        self.mask_path = mask_path
        self.lcut = lcut
        self.delta_ell = delta_ell
        self.l_limber = l_limber
        self.leffs_up = leffs_up
        self.leffs_low = leffs_low
        self.input_dir = input_dir
        self.sigma_z_cut = sigma_z_cut

    def get_bg(self):
        cosmo = ccl.Cosmology(Omega_c=self.Omega_c, Omega_b=self.Omega_b, h=self.h, A_s=self.A_s, n_s=self.n_s)
        f = np.load(self.input_dir+'desilrg_skyregionNS_pzcut'+str(self.sigma_z_cut)+'_zmin0.4_zmax1.1_nbins20/desilrg_beamprofile.fits.npz')
        zs_raw = f['zgrid']
        Ws_raw = f['conv_normed_profiles']
        Ws_interp = sp.interpolate.interp1d(zs_raw, Ws_raw)
        Ws = Ws_interp(self.zs)

        shot_noises = np.load(self.input_dir+'desilrg_skyregionNS_pzcut'+str(self.sigma_z_cut)+'_zmin0.4_zmax1.1_nbins20/desilrg_shotnoises.fits.npy')

        bzs = np.ones_like(self.zs)
        dndzs = [(self.zs, Ws[i]) for i in range(self.nbins)]
        gc_ds = [ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=dndzs[i], bias=(self.zs, bzs), mag_bias=None) for i in range(self.nbins)]

        Cmms = np.array([ccl.angular_cl(cosmo, gc_ds[i], gc_ds[i], self.ells, l_limber = self.l_limber, fkem_chi_min=0) for i in tqdm(range(self.nbins))])

        mask = hp.read_map(self.mask_path)
        mask_apo = mask

        leff_ini = np.arange(self.lcut, self.lmax, self.delta_ell)
        leff_end = leff_ini+self.delta_ell
        b = nmt.NmtBin.from_edges(ell_ini=leff_ini, ell_end=leff_end)
        leffs = b.get_effective_ells()

        fmask = nmt.NmtField(mask_apo, None, spin = 0,n_iter=0, lmax=self.lmax)
        wmask = nmt.NmtWorkspace.from_fields(fmask,fmask,b)
        Bbl = wmask.get_bandpower_windows().squeeze()

        odmaps = np.array([hp.read_map(self.input_dir+'desilrg_skyregionNS_pzcut'+str(self.sigma_z_cut)+'_zmin0.4_zmax1.1_nbins20/desilrg_calibrated_overdensity_map_bin'+str(i+1)+'_nside256.fits') for i in range(self.nbins)])

        fs_data = [nmt.NmtField(mask_apo, [m], n_iter = 0) for m in odmaps]
        pcl_data = np.array([(wmask.decouple_cell(nmt.compute_coupled_cell(fs_data[i], fs_data[i])) - shot_noises[i]) for i in range(len(fs_data))]).squeeze()
        ellmask = (leffs > self.leffs_low) & (leffs < self.leffs_up)

        leffs_masked = leffs[ellmask]
        
        Cmms_bpw = Cmms@Bbl.T
        Cmms_bpw_masked = Cmms_bpw[:, ellmask]
        Cdata_bpw_masked = pcl_data[:, ellmask]

        bg2 = np.sum(Cdata_bpw_masked*Cmms_bpw_masked, axis = 1)/np.sum(Cmms_bpw_masked*Cmms_bpw_masked, axis = 1)

        bg = np.sqrt(bg2)

        for i in range(bg.shape[0]):
            plt.figure()
            plt.loglog(leffs, Cmms_bpw[i]*bg2[i])
            plt.loglog(leffs, pcl_data[i])
            plt.plot()
        
        plt.figure()
        plt.plot(bg)
        np.savez(self.input_dir+'desilrg_skyregionNS_pzcut'+str(self.sigma_z_cut)+'_zmin0.4_zmax1.1_nbins20/'+'desi_lrg_bg_pzcut'+str(self.sigma_z_cut)+'_20bins.npz', bg = bg)

        