import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import fitsio
import healpy as hp
import os
from sklearn.linear_model import LinearRegression
import matplotlib

def gaussian(mean, width, zs):
    return np.exp(-(zs - mean)**2/(2*width**2))

def check_dir(dir):
    if os.path.exists(dir) == False:
        print('directory not found, creating directory...')
        os.mkdir(dir)
        print('directory ', dir, ' is created.')
    else:
        return

class desi_obj:
    def init_desi(self, desi_dir):
        if os.path.exists(desi_dir) == False:
            print('The following directory does not exist, please double check: '+desi_dir)
        else:
            print('configuring to the desi directory: '+desi_dir)
            self.desi_dir = desi_dir

            if os.path.exists(desi_dir + 'dr9_extended_lrg_pzbins.fits'):
                print('found lrg catalog, will be loaded')
                self.desi_catfname = desi_dir + 'dr9_extended_lrg_pzbins.fits'

            if os.path.exists(desi_dir + 'dr9_extended_lrg_pz.fits'):
                print('found associated photo-z catalog, will be loaded')
                self.desi_pzfname = desi_dir + 'dr9_extended_lrg_pz.fits'

            if os.path.exists(desi_dir + 'pixweight-dr7.1-0.22.0_stardens_64_ring.fits'):
                print('found associated pixel weight catalog, will be loaded')
                self.desi_pixweightfname = desi_dir + 'pixweight-dr7.1-0.22.0_stardens_64_ring.fits'

            if os.path.exists(desi_dir + 'imaging_systematics_maps_and_weights_256.fits'):
                print('found associated random catalog, will be loaded')
                self.desi_randcatfname = desi_dir + 'imaging_systematics_maps_and_weights_256.fits'

    def load_catalogue(self, apply_veto_mask = True):
        print('loading catalogue from the desi directory...')
        print('Apply veto mask: ', apply_veto_mask)

        self.random_catalog_nside = 256

        if apply_veto_mask == False:
            self.object_catalog = Table(fitsio.read(self.desi_catfname))
            self.photo_z_catalog = Table(fitsio.read(self.desi_pzfname))
            stardens_64 = fitsio.read(self.desi_pixweightfname)['STARDENS']
            self.stardens = hp.ud_grade(stardens, nside_out=256)

        else:
            object_catalog = Table(fitsio.read(self.desi_catfname))
            photo_z_catalog = Table(fitsio.read(self.desi_pzfname))

            # for now user has to change these mask variables from the code
            min_nobs = 2
            max_ebv = 0.15
            max_stardens = 2500

            island_mask = ~((object_catalog['DEC']<-10.5) & (object_catalog['RA']>120) & (object_catalog['RA']<260))
            nobs_mask = (object_catalog['PIXEL_NOBS_G']>=min_nobs) & (object_catalog['PIXEL_NOBS_R']>=min_nobs) & (object_catalog['PIXEL_NOBS_Z']>=min_nobs)
            lrg_mask = object_catalog['lrg_mask']==0
            ebv_mask = object_catalog['EBV']<max_ebv

            stardens = fitsio.read(self.desi_pixweightfname)  # Stellar density map
            stardens_nside = 64
            stardens_mask = stardens['STARDENS']>=max_stardens
            bad_hp_idx = stardens['HPXPIXEL'][stardens_mask]
            cat_hp_idx = hp.pixelfunc.ang2pix(stardens_nside, object_catalog['RA'], object_catalog['DEC'], lonlat=True, nest=False)
            stardens_mask_good = ~np.in1d(cat_hp_idx, bad_hp_idx)

            joint_mask = island_mask & nobs_mask & lrg_mask & ebv_mask & stardens_mask_good

            self.object_catalog = object_catalog[joint_mask]
            self.photo_z_catalog = photo_z_catalog[joint_mask]
            
            self.object_catalog_keys = self.object_catalog.keys()
            self.photo_z_catalog_keys = self.photo_z_catalog.keys()

            stardens_64 = fitsio.read(self.desi_pixweightfname)['STARDENS']
            stardens_64[bad_hp_idx] = 0
            self.stardens = hp.ud_grade(stardens_64, nside_out=self.random_catalog_nside)

        print('now loading random catalog...')
        self.random_catalog = Table(fitsio.read(self.desi_randcatfname))
        self.random_catalog_keys = self.random_catalog.keys()

        print('no. of objects remaining in the catalog: ', len(self.object_catalog))
        print('object catalog same size as photon-z catalog? ', len(self.object_catalog)==len(self.photo_z_catalog))

        print('object catalog keys: ', self.object_catalog_keys)
        print('photo-z catalog keys: ', self.photo_z_catalog_keys)

        print('random catalog keys: ', self.random_catalog_keys)

    def init_catalog_cuts(self, photosys='NS', sigma_z_max = 1):
        # right now we only support one cut but we can do by object/by redshift bin weighting later
        print('imposing custom cuts to catalog, summary of the cuts:')
        print('survey region: ', photosys)
        print('maximum photo-z error: ', sigma_z_max)

        self.sigma_z_max = sigma_z_max
        self.photosys = photosys

        if photosys == 'NS':
            self.custom_mask = (self.photo_z_catalog['Z_PHOT_STD']/(1+self.object_catalog['Z_PHOT_MEDIAN'])) < self.sigma_z_max
        else:
            photosys_mask = self.object_catalog['PHOTSYS'] == self.photosys
            pz_mask = (self.photo_z_catalog['Z_PHOT_STD']/(1+self.object_catalog['Z_PHOT_MEDIAN'])) < self.sigma_z_max
            self.custom_mask = photosys_mask & pz_mask
            
        self.object_catalog = self.object_catalog[self.custom_mask]
        self.photo_z_catalog = self.photo_z_catalog[self.custom_mask]
        
        print('custom cuts completed.')

    def get_binned_catalog(self, zmin, zmax, nbins):
        print('getting binned catalogs, summary to the binning:')
        print('minimum redshift: ', zmin)
        print('maximum redshift: ', zmax)
        print('number of bins: ', nbins)

        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins
        
        zbins = np.linspace(zmin, zmax, nbins+1)
        zbins_low = zbins[:-1]
        zbins_up = zbins[1:]

        self.zbins = zbins
        self.zbins_low = zbins_low
        self.zbins_up = zbins_up

        photo_z_bin = []
        ra_bin = []
        dec_bin = []
        photo_z_err_bin = []
        photo_z_median_bin = []

        for i in range(nbins):
            bin_filter = (self.object_catalog['Z_PHOT_MEDIAN'] >= zbins_low[i]) & (self.object_catalog['Z_PHOT_MEDIAN'] < zbins_up[i])            
            
            photo_z_bin.append(self.object_catalog['Z_PHOT_MEDIAN'][bin_filter])
            ra_bin.append(self.object_catalog['RA'][bin_filter])
            dec_bin.append(self.object_catalog['DEC'][bin_filter])
            photo_z_err_bin.append(self.photo_z_catalog['Z_PHOT_STD'][bin_filter]/(1+self.object_catalog['Z_PHOT_MEDIAN'])[bin_filter])
            photo_z_median_bin.append(np.median(self.object_catalog['Z_PHOT_MEDIAN'][bin_filter]))
            
        self.photo_z_bin = photo_z_bin
        self.ra_bin = ra_bin
        self.dec_bin = dec_bin
        self.photo_z_err_bin = photo_z_err_bin
        self.photo_z_median_bin = np.array(photo_z_median_bin)

        print('catalog binning completed.')

    def get_mask(self, mask_path):
        status = os.path.exists(mask_path)
        self.mask_path = mask_path
        
        print('check if mask directory exits: ', status)
        
        if status == True:
            print('loading mask and converting to fsky...')
            self.mask = hp.read_map(self.mask_path).astype(bool)
            self.mask_nside = hp.npix2nside(self.mask.shape[0])
            self.fsky = np.mean(self.mask**2)
    
    def get_shot_noises(self):
        
        shot_noises = []
        ngals = []
        
        for i in range(len(self.photo_z_bin)):
            ngals.append(self.photo_z_bin[i].shape[0])
            ngal_fsky = self.photo_z_bin[i].shape[0]/self.fsky
            shotnoise = np.pi*4/ngal_fsky
            shot_noises.append(shotnoise)
            
        self.shotnoises = np.array(shot_noises)
        self.ngals = np.array(ngals)

    def get_photo_z_err_median_bin(self, type = 'zhou'):
        print('getting equivalent photo-z error for each bin')
        print('type of photo-z: ', type)
        
        if type == 'zhou':
            self.photo_z_err_median_bin = 0.027*(1+self.photo_z_median_bin)

        if type == 'from_cuts':

            photo_z_err_median_bin = []
            
            for i in range(len(self.photo_z_err_bin)):
                photo_z_err_median_bin.append(np.median(self.photo_z_err_bin[i]))
            
            self.photo_z_err_median_bin = np.array(photo_z_err_median_bin)

    def get_redshift_grid(self, auto=True, zgrid_min=None, zgrid_max=None, ngrid=None):
        if auto == True:
            # this should be checked with maximum photo-z error but defer to future
            self.z_extend = 4*(self.zmax-self.zmin)/5
            self.zgrid_min = self.zmin-self.z_extend
            self.zgrid_max = self.zmax+self.z_extend
            self.ngrid = 1101
            
            self.zgrid = np.linspace(self.zmin-self.z_extend, self.zmax+self.z_extend, self.ngrid)
        else:
            self.zgrid_min = zgrid_min
            self.zgrid_max = zgrid_max
            self.ngrid = ngrid
            
            self.zgrid = np.linspace(self.zgrid_min, self.zgrid_max, self.ngrid)

    def get_unconvolved_normalized_profiles(self):
        profiles = []
        for i in range(self.nbins):
            hist, edges = np.histogram(self.photo_z_bin[i], self.zgrid)
            profile = hist/np.trapz(hist, self.zgrid[1:])
            profiles.append(profile)
        
        self.unconvolved_normalized_profiles = profiles
        self.profile_edges = edges

    def get_convolved_normalized_profiles(self):
        profiles = []
        
        for i in range(self.nbins):
            shape = self.unconvolved_normalized_profiles[i]
            std = self.photo_z_err_median_bin[i]
            gaus = gaussian((self.zgrid[0]+self.zgrid[-1])/2, std, self.zgrid)
            profile = np.convolve(gaus, shape, 'same')
            profiles.append(profile/np.trapz(profile, self.zgrid))

        self.convolved_normalized_profiles = profiles

    def get_diagnosis_plot(self, save_dir=None):
        cmap = matplotlib.colormaps['cividis']
        colors = cmap(np.linspace(0,1,self.nbins))
        nrows = 2
        ncols = 3

        # if self.nbins > 4:
        #     idxs = np.linspace(0, self.nbins-1, 5).astype(int)
        # else:
        #     idxs = np.arange(self.nbins)
        
        fig, axs = plt.subplots(nrows=nrows, ncols = ncols, figsize=(7*ncols,6*nrows))
        fig.subplots_adjust(hspace=0.3)
        fig.subplots_adjust(wspace=0.2)
    
        # number of galaxies
        ax1 = axs[0,0]
        
        ax1.tick_params(axis='both', which='major',labelsize=18, length = 10, width = 1, direction='in', right='on', top='on')
        ax1.tick_params(axis='both', which='minor',labelsize=16, length = 5, width = 1, direction='in', right='on', top='on')
    
        ax1.plot(np.arange(self.nbins)+1, self.ngals)
        ax1.set_xlabel('redshift bin', size=16)
        ax1.set_ylabel('number of galaxies', size = 16)
        ax1.grid()
        
        # shotnoises
        ax2 = axs[0,1]
        ax2.tick_params(axis='both', which='major',labelsize=18, length = 10, width = 1, direction='in', right='on', top='on')
        ax2.tick_params(axis='both', which='minor',labelsize=16, length = 5, width = 1, direction='in', right='on', top='on')
    
        ax2.plot(np.arange(self.nbins)+1, self.shotnoises)
        ax2.set_xlabel('redshift bin', size=16)
        ax2.set_ylabel(r'shotnoise (sr)', size = 16)
        ax2.grid()
    
        ax3 = axs[0,2]
        ax3.tick_params(axis='both', which='major',labelsize=18, length = 10, width = 1, direction='in', right='on', top='on')
        ax3.tick_params(axis='both', which='minor',labelsize=16, length = 5, width = 1, direction='in', right='on', top='on')
        
        # profile shapes, unormalized
        edge_width = self.profile_edges[1] - self.profile_edges[0]
        bin_width = self.zbins[1] - self.zbins[0]
        per_sqdeg = 1/46800
        conv_factor = per_sqdeg/self.fsky*edge_width/bin_width
        
        for i in range(self.nbins):
            ngal_conv = self.ngals[i]*conv_factor
            ax3.stairs(ngal_conv*self.unconvolved_normalized_profiles[i], self.profile_edges, color = colors[i],label = 'bin'+str(i+1))
            ax3.stairs(ngal_conv*self.convolved_normalized_profiles[i][:-1], self.zgrid, color = colors[i])
            # ax3.vlines(self.photo_z_median_bin, -1, 10)
    
        ax3.set_ylabel('galaxy density per square degree', size = 13)
        ax3.set_xlabel('redshift', size = 16)
        ax3.grid()
    
        ax3.legend(prop={'size':14})
    
        ax4 = axs[1,0]
        ax4.tick_params(axis='both', which='major',labelsize=18, length = 10, width = 1, direction='in', right='on', top='on')
        ax4.tick_params(axis='both', which='minor',labelsize=16, length = 5, width = 1, direction='in', right='on', top='on')

        # only work for desilrg sample
        photo_z_err_grid = np.linspace(0, 0.14, 200)
        
        for i in range(self.nbins):
            hist, edges = np.histogram(self.photo_z_err_bin[i], photo_z_err_grid)
            ax4.stairs(hist, edges, label = 'bin'+str(i+1))
            
        ax4.legend(prop={'size':14})
        ax4.set_xlabel('photo-z err', size=16)
        ax4.set_ylabel('counts', size=16)
        ax4.ticklabel_format(axis='y', scilimits = (0,0))
        ax4.grid()
    
        ax5 = axs[1,1]
        ax5.tick_params(axis='both', which='major',labelsize=18, length = 10, width = 1, direction='in', right='on', top='on')
        ax5.tick_params(axis='both', which='minor',labelsize=16, length = 5, width = 1, direction='in', right='on', top='on')
        ax5.ticklabel_format(axis='y', scilimits = (0,0))
    
        ax5.plot(np.arange(self.nbins)+1, self.photo_z_err_median_bin, label = 'after photo-z cut, estimate from data')
        ax5.plot(np.arange(self.nbins)+1, 0.027*(1+self.photo_z_median_bin), '--', label = 'zhou')
        ax5.set_xlabel('redshift bins', size=16)
        ax5.set_ylabel('photo-z err median', size = 16)
        ax5.grid()
        ax5.legend(prop={'size':14})

        if save_dir == None:
            plt.savefig('diagnosis.pdf', bbox_inches='tight')
        else:
            plt.savefig(save_dir + 'diagnosis.pdf', bbox_inches='tight')
        plt.show()

    def initialize_imaging_properties(self, mask = None):
        self.full_imaging_properties = ['FRACAREA', 'n_randoms', 'EBV', 'PSFDEPTH_W1', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
        'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'STARDENS']

        print('Availble properties:')
        for i in range(len(self.full_imaging_properties)-2):
            print('index '+str(i+2)+': ', self.full_imaging_properties[i+2])
        print('you can select them by calling this function again with a boolean mask as argument (0 means not using), otherwise all of them will be used')

        if mask is None:
            self.applied_imaging_properties = self.full_imaging_properties
        else:
            self.applied_imaging_properties = list(np.array(self.full_imaging_properties)[mask])

        print('Currently applied imaging properties: ', self.applied_imaging_properties)

    def get_systematic_maps(self):
        print('getting systematic maps for regression...')

        sysmaps = np.zeros(((len(self.applied_imaging_properties)), hp.nside2npix(256)))
        pixpos = self.random_catalog['HPXPIXEL']
        
        for i in range(sysmaps.shape[0]):
            prop = self.applied_imaging_properties[i]
            if prop == 'STARDENS':
                sysmaps[i, pixpos] = self.stardens[pixpos]
            else:
                sysmaps[i, pixpos] = self.random_catalog[prop]

        self.systematic_maps = np.array(sysmaps)
        print('systematic maps are constructed')

    def get_dmaps(self, nside=256):
        self.dmaps_nside = nside
        dmaps = np.zeros((self.nbins, hp.nside2npix(nside)))
        
        for i in range(self.nbins):
            pixbin = hp.ang2pix(nside=nside, phi = self.ra_bin[i]*np.pi/180, theta = np.pi/2 - self.dec_bin[i]*np.pi/180)

            for j in range(pixbin.shape[0]):
                dmaps[i, pixbin[j]] += 1

        self.dmaps = dmaps

    def get_odmaps(self):
        # check whether the nside of mask matches with maps
        if self.mask_nside != self.dmaps_nside:
            print('mask nside and galaxy density maps nside mismatch, please reload either of them.')
            return
            
        naive_odmaps = np.zeros((self.nbins, hp.nside2npix(self.dmaps_nside)))
        
        for i in range(naive_odmaps.shape[0]):
            print('making galaxy overdensity for redshift bin', str(i+1))
            dbar = np.sum(self.dmaps[i])/np.sum(self.mask)
            dbar_map = dbar*self.mask
            good = self.mask > 0
            naive_odmaps[i, good] = self.dmaps[i, good]/dbar_map[good] - 1.

        self.odmaps = naive_odmaps

    def get_calibrated_odmaps(self):
        
        if self.mask_nside != self.dmaps_nside:
            print('mask nside and galaxy density maps nside mismatch, please reload either of them.')
            return
        if self.mask_nside != self.random_catalog_nside:
            print('mask nside and random map nside mismatch, please reload either of them.')
            return
        if self.random_catalog_nside != self.dmaps_nside:
            print('random map nside and galaxy density maps nside mismatch, please reload either of them.')
            return

        self.dmaps_calibrated_nside = self.dmaps_nside
            
        odmaps_calibrated = np.zeros((self.nbins, hp.nside2npix(self.dmaps_nside)))
        coeffs = []
        intercepts = []
        
        for i in range(self.nbins):
            print('making calibrated galaxy overdensity for redshift bin', str(i+1))
            reg = LinearRegression()
            reg.fit(self.systematic_maps[:, self.mask].T, self.dmaps[i, self.mask])

            coeff = reg.coef_
            intercept = reg.intercept_
            coeffs.append(coeff)
            intercepts.append(intercept)

            dmap_prediced_from_radom = (intercept + coeff@self.systematic_maps[:, self.mask])

            normalization = np.sum(dmap_prediced_from_radom)/np.sum(self.dmaps[i, self.mask])
            odmaps_calibrated[i, self.mask] = self.dmaps[i, self.mask]/(dmap_prediced_from_radom)*normalization -1.

        self.odmaps_calibrated = odmaps_calibrated

    def output(self, type = 'dmap', dir = None):
        print('output type: ', type)
        
        dir_tag = 'desilrg_skyregion'+self.photosys+'_pzcut'+str(np.round(self.sigma_z_max, 3))+'_zmin'+str(np.round(self.zmin,3))+'_zmax'+str(np.round(self.zmax,3))+'_nbins'+str(self.nbins)+'/'
        
        if dir is not None:
            dir_full = dir+dir_tag
            print('you decided to save the output to: ', dir_full)

        check_dir(dir_full)
        
        if type == 'dmap':
            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_density_map_bin'+str(i+1)+'_nside'+str(self.dmaps_nside)+'.fits'
                hp.write(fname, self.dmaps[i])

            print('output complete.')
            
        if type == 'odmap':
            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_overdensity_map_bin'+str(i+1)+'_nside'+str(self.dmaps_nside)+'.fits'
                hp.write(fname, self.odmaps[i])

            print('output complete.')
        
        if type == 'calibrated_odmap':
            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_calibrated_overdensity_map_bin'+str(i+1)+'_nside'+str(self.dmaps_calibrated_nside)+'.fits'
                hp.write(fname, self.odmaps_calibrated[i])

            print('output complete.')

        if type == 'beam_profile':
            fname = dir_full + 'desilrg_beamprofile.fits'
            np.savez(fname, zgrid = self.zgrid, conv_normed_profiles = self.convolved_normalized_profiles, unconv_normed_profiles = self.unconvolved_normalized_profiles, profile_edges = self.profile_edges)
        
        if type == 'shotnoises':
            fname = dir_full + 'desilrg_shotnoises.fits'
            np.save(fname, self.shotnoises)

        if type == 'all':
            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_density_map_bin'+str(i+1)+'_nside'+str(self.dmaps_nside)+'.fits'
                hp.write_map(fname, self.dmaps[i])

            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_overdensity_map_bin'+str(i+1)+'_nside'+str(self.dmaps_nside)+'.fits'
                hp.write_map(fname, self.odmaps[i])

            for i in range(self.nbins):
                print('outputing map ', i+1)
                fname = dir_full + 'desilrg_calibrated_overdensity_map_bin'+str(i+1)+'_nside'+str(self.dmaps_calibrated_nside)+'.fits'
                hp.write_map(fname, self.odmaps_calibrated[i])

            fname = dir_full + 'desilrg_beamprofile.fits'
            np.savez(fname, zgrid = self.zgrid, zbins = self.zbins, conv_normed_profiles = self.convolved_normalized_profiles, unconv_normed_profiles = self.unconvolved_normalized_profiles, profile_edges = self.profile_edges, photo_z_err_bin = self.photo_z_err_median_bin)

            fname = dir_full + 'desilrg_shotnoises.fits'
            np.save(fname, self.shotnoises)

            print('output complete.')