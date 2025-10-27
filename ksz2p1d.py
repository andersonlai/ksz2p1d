# baseline import
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pyccl as ccl
import scipy as sp
from tqdm import tqdm

# load modules
import desi_utils
from desi_utils import check_dir
import fiducial_utils as fid_utils
import qe
# import coadd
import get_bias
from mfqml import *
from utilities import *
import opt_einsum as oe

# interfacing with qmlfast?
def get_pix_cov_block(cl,Pl_ij,lmin,lmax):
    cov = oe.contract('ijk,i->jk',Pl_ij[lmin:lmax],cl[lmin:lmax])
    #cov += 1e3*oe.contract('ijk,i->jk',Pl_ij[:lmin],np.ones(lmin))
    return cov

class ksz2p1d_obj:
    def init(self, output_dir):
        check_dir(output_dir)
        self.output_dir = output_dir
        
    def init_desi(self, sigma_z_cut = 1, photosys = 'NS',
                  pz_type = 'zhou', zmin=0.4, zmax=1.1, nbins=20, zgrid_min=0.2, zgrid_max=1.3,
                  ngrid=1101,
                  desi_input_dir = '/scratch2/acmlai/ACTDR6/ksz2p1d_test/desi_inputs/',
                  mask_path1 = '/scratch2/acmlai/ACTDR6/ksz2p1d_test/desi_inputs/lrg_s01_msk.hpx2048.fits',
                  mask_path_cal='/scratch2/acmlai/ACTDR6/masks/desi_mask_nside256.fits',
                  mask_path_output='/scratch2/acmlai/ACTDR6/masks/desi_mask_nside4096_binary.fits'):
        
        self.sigma_z_cut = sigma_z_cut
        self.photosys = photosys
        self.pz_type = pz_type
        self.desi_output_dir = self.output_dir
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins
        self.zgrid_min = zgrid_min
        self.zgrid_max = zgrid_max
        self.ngrid = ngrid
        self.desi_input_dir = '/scratch2/acmlai/ACTDR6/ksz2p1d_test/desi_inputs/'

        self.mask_path1 = mask_path1
        self.mask_path_cal = mask_path_cal
        self.mask_path_output = mask_path_output

    def run_desi(self):
        desi_obj = desi_utils.desi_obj()
        desi_obj.init_desi(self.desi_input_dir)
        desi_obj.load_catalogue()
        desi_obj.init_catalog_cuts(photosys=self.photosys, sigma_z_max = self.sigma_z_cut)
        #baseline flow
        desi_obj.get_mask(self.mask_path1)
        desi_obj.get_binned_catalog(self.zmin, self.zmax, self.nbins)
        desi_obj.get_shot_noises()
        desi_obj.get_photo_z_err_median_bin(type=self.pz_type)
        desi_obj.get_redshift_grid(auto = False, zgrid_min=self.zgrid_min, zgrid_max=self.zgrid_max, ngrid=self.ngrid)
        desi_obj.get_unconvolved_normalized_profiles()
        desi_obj.get_convolved_normalized_profiles()
        desi_obj.get_diagnosis_plot()
        desi_obj.initialize_imaging_properties()

        # photosystematics
        imaging_mask = np.ones(14)
        # imaging_mask[[13]] = 0
        imaging_mask = imaging_mask.astype(bool)
        desi_obj.initialize_imaging_properties(mask=imaging_mask)
        desi_obj.get_systematic_maps()
        desi_obj.get_dmaps()

        mask_path = self.mask_path_cal
        desi_obj.get_mask(mask_path)
        desi_obj.get_calibrated_odmaps()

        mask_path = self.mask_path_output
        desi_obj.get_mask(mask_path)
        desi_obj.get_dmaps(nside=4096)
        desi_obj.get_odmaps()
        
        desi_obj.output(type = 'all', dir = self.output_dir)
        self.desi_dir = self.output_dir + 'desilrg_skyregion'+self.photosys+'_pzcut'+str(self.sigma_z_cut)+'_zmin'+str(self.zmin)+'_zmax'+str(self.zmax)+'_nbins'+str(self.nbins)+'/'

    def run_bias(self):
        bias_obj = get_bias.get_biases()
        
        bias_obj.init(nbins = self.nbins, nside = 256, zs = np.linspace(0.2, 1.25, 500), Omega_c=0.268, Omega_b=0.048, h=0.675, A_s=2.1e-9, n_s=0.96,
             mask_path = '/scratch2/acmlai/ACTDR6/masks/desi_mask_nside256.fits',
             lcut = 6, delta_ell = 6, l_limber = 120, leffs_up=110, leffs_low=80,
             input_dir=self.output_dir, sigma_z_cut = self.sigma_z_cut)
        
        bias_obj.get_bg()

    def run_Cgtau_fiducials(self):
        fid = fid_utils.fiducial()
        fid.initialize_profile(self.desi_dir+'desilrg_beamprofile.fits.npz')
        fid.inititalize_hod_parameters()
        fid.initialize_electron_profile()
        fid.get_fiducials()

        bg = np.load(self.desi_dir+'desi_lrg_bg_pzcut'+str(self.sigma_z_cut)+'_20bins.npz')['bg']

        np.savez(self.desi_dir+'fiducials.npz', 
        Cmm1hs = fid.Cmm1hs, Cmm2hs = fid.Cmm2hs,
        Cmtau1hs = fid.Cmtau1hs, Cmtau2hs = fid.Cmtau2hs,
        Cgg1hs = fid.Cgg1hs, Cgg2hs = fid.Cgg2hs,
        Cgtau1hs = fid.Cgtau1hs, Cgtau2hs = fid.Cgtau2hs,
        bg=bg)

    def init_qe(self, Tmask_fname, gmask_fname, overlap_in_fname, overlap_out_fname, Tmap_dir, lmax = 10000, nside = 4096, nside_out = 128, lcutoff = 300):
        self.qe_lmax = lmax
        self.qe_nside = nside
        self.qe_nside_out = nside_out
        self.qe_lcutoff = lcutoff

        self.Tmask_fname = Tmask_fname
        self.gmask_fname = gmask_fname

        self.overlap_in_fname = overlap_in_fname
        self.overlap_out_fname = overlap_out_fname

        self.Tmap_dir = Tmap_dir

    def run_qe(self, nilc=False):
        estim = qe.estimator()
        estim.initialize(self.qe_lmax, self.qe_nside)
        estim.load_masks(self.Tmask_fname, self.gmask_fname, self.overlap_in_fname, self.overlap_out_fname)

        estim.load_data(self.Tmap_dir, self.desi_dir)
        
        shotnoises = np.load(self.desi_dir+'desilrg_shotnoises.fits.npy')

        estim.load_Cgtau(self.desi_dir)

        estim.get_estimator(self.qe_nside_out, self.qe_lcutoff)
        
        if nilc==True:
            np.savez(self.output_dir+'QE_output.npz',
                     vmaps_f090 = estim.vmaps_dict['f090'], vmaps_f150 = estim.vmaps_dict['f150'],
                     vmaps_meansub_f090 = estim.vmaps_meansub_dict['f090'],
                     vmaps_meansub_f150 = estim.vmaps_meansub_dict['f150'],
                     Nvvs_f090 = estim.Nvvs_dict['f090'], Nvvs_f150 = estim.Nvvs_dict['f150'],
                    vmaps_nilc = estim.vmaps_dict['nilc'], vmaps_meansub_nilc = estim.vmaps_meansub_dict['nilc'],
                     Nvvs_nilc = estim.Nvvs_dict['nilc'])
        else:
             np.savez(self.output_dir+'QE_output.npz',
                     vmaps_f090 = estim.vmaps_dict['f090'], vmaps_f150 = estim.vmaps_dict['f150'],
                     vmaps_meansub_f090 = estim.vmaps_meansub_dict['f090'],
                     vmaps_meansub_f150 = estim.vmaps_meansub_dict['f150'],
                     Nvvs_f090 = estim.Nvvs_dict['f090'], Nvvs_f150 = estim.Nvvs_dict['f150'])

    def get_coadd(self, llow, lup, nilc=False):
        self.Nvv_llow = llow
        self.Nvv_lup = lup
        
        f = np.load(self.output_dir+'QE_output.npz')
        vmaps_meansub_f090 = f['vmaps_meansub_f090']
        vmaps_meansub_f150 = f['vmaps_meansub_f150']
        Nvvs_f090 = f['Nvvs_f090']
        Nvvs_f150 = f['Nvvs_f150']
        
        mask = hp.read_map('/scratch2/acmlai/ACTDR6/masks/overlap_ACTDR6nilc_tszmasked_desilrg_binary_nside128.fits')
        mask_apo = nmt.mask_apodization(mask, 0.3, apotype = 'Smooth')
        hp.mollview(mask_apo)
        
        nside = 128
        lmax = 3*nside - 1
        
        lcut = 3
        delta_ell = 3
        leff_ini = np.arange(lcut, lmax, delta_ell)
        leff_end = leff_ini+delta_ell
        b = nmt.NmtBin.from_edges(ell_ini=leff_ini, ell_end=leff_end)
        leffs = b.get_effective_ells()
        
        fmask = nmt.NmtField(mask_apo, None, spin = 0,n_iter=0, lmax=lmax)
        wmask = nmt.NmtWorkspace.from_fields(fmask,fmask,b)
        Bbl = wmask.get_bandpower_windows().squeeze()
        
        fs_f090 = [nmt.NmtField(mask_apo, [m], n_iter = 0) for m in vmaps_meansub_f090]
        fs_f150 = [nmt.NmtField(mask_apo, [m], n_iter = 0) for m in vmaps_meansub_f150]
        
        pcl_Cvvs_f090 = np.array([wmask.decouple_cell(nmt.compute_coupled_cell(fs_f090[i], fs_f090[i])) for i in range(len(fs_f090))]).squeeze()
        pcl_Cvvs_f150 = np.array([wmask.decouple_cell(nmt.compute_coupled_cell(fs_f150[i], fs_f150[i])) for i in range(len(fs_f150))]).squeeze()
        pcl_Cvvs_f090xf150 = np.array([wmask.decouple_cell(nmt.compute_coupled_cell(fs_f090[i], fs_f150[i])) for i in range(len(fs_f150))]).squeeze()
        
        llow = self.Nvv_llow
        lup = self.Nvv_lup
        leff_mask = (leffs > llow) & (leffs < lup)
        
        Nvvs_f090_pcl_mean = np.mean(pcl_Cvvs_f090[:, leff_mask], axis = 1)
        Nvvs_f150_pcl_mean = np.mean(pcl_Cvvs_f150[:, leff_mask], axis = 1)
        Nvvs_f090xf150_pcl_mean = np.mean(pcl_Cvvs_f090xf150[:, leff_mask], axis = 1)
        
        A = (Nvvs_f150_pcl_mean - Nvvs_f090xf150_pcl_mean)/(Nvvs_f150_pcl_mean + Nvvs_f090_pcl_mean - 2*Nvvs_f090xf150_pcl_mean)
        B = (Nvvs_f090_pcl_mean - Nvvs_f090xf150_pcl_mean)/(Nvvs_f150_pcl_mean + Nvvs_f090_pcl_mean - 2*Nvvs_f090xf150_pcl_mean)
        
        vmaps_meansub_coadd = A[:,None]*vmaps_meansub_f090 + B[:,None]*vmaps_meansub_f150
        
        Nvvs_coadd_mean = A**2*Nvvs_f090_pcl_mean + B**2*Nvvs_f150_pcl_mean + 2*A*B*Nvvs_f090xf150_pcl_mean

        if nilc == True:
            vmaps_meansub_nilc = f['vmaps_meansub_nilc']
            fs_nilc = [nmt.NmtField(mask_apo, [m], n_iter = 0) for m in vmaps_meansub_nilc]
            pcl_Cvvs_nilc = np.array([wmask.decouple_cell(nmt.compute_coupled_cell(fs_nilc[i], fs_nilc[i])) for i in range(len(fs_nilc))]).squeeze()
            Nvvs_nilc_pcl_mean = np.mean(pcl_Cvvs_nilc[:, leff_mask], axis = 1)
            
            np.savez(self.output_dir+'coadded_output.npz',
                 Nvvs_f090_pcl_mean=Nvvs_f090_pcl_mean, Nvvs_f150_pcl_mean=Nvvs_f150_pcl_mean,
                 Nvvs_f090xf150_pcl_mean=Nvvs_f090xf150_pcl_mean,
                 Nvvs_coadd_mean=Nvvs_coadd_mean,
                vmaps_meansub_coadd = vmaps_meansub_coadd,
                    Nvvs_nilc_pcl_mean= Nvvs_nilc_pcl_mean)
        else:
            np.savez(self.output_dir+'coadded_output.npz',
                     Nvvs_f090_pcl_mean=Nvvs_f090_pcl_mean, Nvvs_f150_pcl_mean=Nvvs_f150_pcl_mean,
                     Nvvs_f090xf150_pcl_mean=Nvvs_f090xf150_pcl_mean,
                     Nvvs_coadd_mean=Nvvs_coadd_mean,
                    vmaps_meansub_coadd = vmaps_meansub_coadd)

    
    def get_qml_fiducials(self):
        cosmo = ccl.Cosmology(Omega_c=0.268, Omega_b=0.048, h=0.675, A_s=2.1e-9, n_s=0.96)
    
        zs = np.linspace(0.2, 1.25, 2000)
        
        f = np.load(self.desi_dir+'desilrg_beamprofile.fits.npz')
        zs_raw = f['zgrid']
        Ws_raw = f['conv_normed_profiles']
        Ws_interp = sp.interpolate.interp1d(zs_raw, Ws_raw)
        Ws = Ws_interp(zs)
        
        ks = np.logspace(-5, 2, int(256))
        H0 = 0.675*100
        lmax = 120
        l_limber_max =lmax+1
        ells = np.arange(lmax+1)
        
        a = np.flip(1/(1+zs))
        chi = ccl.comoving_radial_distance(cosmo, a)
        
        fz = ccl.growth_rate(cosmo, a)
        Hs = ccl.h_over_h0(cosmo, a)*H0
        
        faH = fz*Hs*a/3e5
        
        transfer_a = (a, faH)
        transfer_k = (np.log(ks), 1./ks)
        
        velocity_tracers = []
        
        for i in range(self.nbins):
            custom_tracer = ccl.Tracer()
            kernel = ccl.get_density_kernel(cosmo, dndz=(zs, Ws[i]))
            custom_tracer.add_tracer(cosmo, kernel=kernel, transfer_a=transfer_a, transfer_k=transfer_k, der_bessel=1)
            velocity_tracers.append(custom_tracer)
        
        galaxy_tracers = [ccl.NumberCountsTracer(cosmo, has_rsd=True, dndz=(zs, Ws[i]), bias=(zs, np.ones_like(zs))) for i in range(self.nbins)]
        
        Cggs = np.array([ccl.angular_cl(cosmo, galaxy_tracers[i], galaxy_tracers[j], ells, l_limber = l_limber_max, fkem_chi_min=0) for i in tqdm(range(self.nbins)) for j in range(self.nbins) if j >= i])
        
        Cvvs = np.array([ccl.angular_cl(cosmo, velocity_tracers[i], velocity_tracers[j], ells, l_limber=l_limber_max, fkem_chi_min=0, fkem_Nchi=int(1e5)) for i in tqdm(range(self.nbins)) for j in range(self.nbins) if j >= i])
        
        Cgvs = np.array([ccl.angular_cl(cosmo, galaxy_tracers[i], velocity_tracers[j], ells, l_limber=l_limber_max, fkem_chi_min=0, fkem_Nchi=int(1e5)) for i in tqdm(range(self.nbins)) for j in range(self.nbins)])
        
        np.savez(self.output_dir+'pyccl_fidcuial_20bins_pzcut'+str(self.sigma_z_cut)+'.npz', Cvvs = Cvvs, Cggs = Cggs, Cgvs = Cgvs)
        # np.savez(self.output_dir+'pyccl_fidcuial_20bins_pzcut'+str(self.sigma_z_cut)+'.npz', Cggs = Cggs)

    def get_invcov(self, nside_out=32, lcut=5, lmax=115, ksznoise_type='coadd', rescale = False):
        f_fid = np.load(self.output_dir+'pyccl_fidcuial_20bins_pzcut'+str(self.sigma_z_cut)+'.npz')
        Nf = self.nbins
        mask = hp.read_map('/scratch2/acmlai/ACTDR6/masks/overlap_ACTDR6nilc_tszmasked_desilrg_binary_nside32.fits')
        Cdds = f_fid['Cggs']
        bias = np.load(self.desi_dir+'desi_lrg_bg_pzcut'+str(self.sigma_z_cut)+'_20bins.npz')['bg']
        shotnoises = np.load(self.desi_dir+'desilrg_shotnoises.fits.npy')
        Cvvs = f_fid['Cvvs']
        if ksznoise_type == 'coadd':
            ksznoises = np.load(self.output_dir+'coadded_output.npz')['Nvvs_coadd_mean']
        elif ksznoise_type == 'nilc':
            ksznoises = np.load(self.output_dir+'coadded_output.npz')['Nvvs_nilc_pcl_mean']
        if rescale == True:
            A = np.load(self.output_dir+'rescaling.npy')
            ksznoises /= A**2

        theta, phi = theta_phi_my(mask)
        Pl_ij = get_Pl_ij_my(mask, lmax=lmax)
        omega_pix = 4 * np.pi / hp.nside2npix(nside_out)
        Np = int(np.sum(mask))
        
        Z, pi = construct_Z_and_Pi(theta,phi,lmax,lcut)
        eta=100
        
        Cggs = []
        idx = 0
        for i in range(Nf):
            for j in range(Nf):
                if j >= i:
                    Cggs.append(Cdds[idx, :lmax+1]*bias[i]*bias[j])
                    # print(idx)
                    idx += 1
        
        clths_clean = Cggs
        noises = shotnoises
        
        C_map = np.ones((Nf, Nf))

        large_cov = np.zeros([Nf*Np,Nf*Np])
        
        for n, (i,j) in enumerate(np.array(np.triu_indices(Nf)).T):
            block = get_pix_cov_block(clths_clean[n],Pl_ij,0,lmax)
            if i==j:
                # block += np.identity(Np)*noises[i]/omega_pix
                block += np.identity(Np)*noises[i]/omega_pix
            block = block*C_map[i,j]    
            large_cov[block_np(i,j,Np)] = block
            if i != j:
                large_cov[block_np(j,i,Np)] = block.T
        
        M=np.linalg.inv(large_cov+np.kron(np.identity(Nf),eta*Z@Z.T))
        for i in tqdm(range(Nf)):
            for j in range(Nf):
                if i==j:
                    b = M[i*Np:(i+1)*Np,j*Np:(j+1)*Np]
                    M[i*Np:(i+1)*Np,j*Np:(j+1)*Np] = pi@b@pi.T
        
        np.save(self.output_dir+'Mgg_lmax'+str(lmax)+'_lcut'+str(lcut)+'.npz', M)

        Cvvs = f_fid['Cvvs']
        clths_clean = list(Cvvs[:,:lmax+1])
        noises = ksznoises
        
        large_cov = np.zeros([Nf*Np,Nf*Np])
        
        for n, (i,j) in enumerate(np.array(np.triu_indices(Nf)).T):
            block = get_pix_cov_block(clths_clean[n],Pl_ij,0,lmax)
            if i==j:
                # block += np.identity(Np)*noises[i]/omega_pix
                block += np.identity(Np)*noises[i]/omega_pix
            block = block*C_map[i,j]    
            large_cov[block_np(i,j,Np)] = block
            if i != j:
                large_cov[block_np(j,i,Np)] = block.T
        
        Z, pi = construct_Z_and_Pi(theta,phi,lmax,lcut)
        
        eta=100
        
        M=np.linalg.inv(large_cov+np.kron(np.identity(Nf),eta*Z@Z.T))
        for i in tqdm(range(Nf)):
            for j in range(Nf):
                if i==j:
                    b = M[i*Np:(i+1)*Np,j*Np:(j+1)*Np]
                    M[i*Np:(i+1)*Np,j*Np:(j+1)*Np] = pi@b@pi.T
        if rescale == True:
            np.save(self.output_dir+'Mvv_lmax'+str(lmax)+'_lcut'+str(lcut)+'_'+ksznoise_type+'ksznoise_rescaled.npz', M)
        else:
            np.save(self.output_dir+'Mvv_lmax'+str(lmax)+'_lcut'+str(lcut)+'_'+ksznoise_type+'ksznoise.npz', M)
 

            