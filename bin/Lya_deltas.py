import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as scpo
import os

from astropy.io import fits as pyfits
from astropy.table import Table
import fitsio

# Desi catalog test
import desispec.io
from desispec.interpolation import resample_flux

import glob
from itertools import compress
import healpy

import argparse

import empca


# def. of comovil distance calculation
def Rcomv(lam_):
   z = ( lam_ )/1216 - 1
   return np.interp(z, z_, r_)

# def. conversion of r, declination and ra to x y z
def coordC(r,ra,dec):
   return r*np.cos(dec)*np.cos(ra), r*np.cos(dec)*np.sin(ra), r*np.sin(dec)

# def. Function to minimize the error of the linear 
# regression for the continuum fit (the most basic fit)
def chi2( alpha, *args ):
   a,b = alpha
   w,flux,ivar = args
   return np.sum( (( flux - (a*w+b) )**2 ) * ivar )

# To crop the DESI spectra filenames for healpy
def splitID(fi_str):
   fi_str= fi_str.split("spectra-16-")[1]
   return fi_str.split(".fits")[0]

# Normalize spectra with a scale factor to get spectra amplitude ~ 1
def normalizeSpec( x, y, xmin, xmax, normfactor = 8):
   xrange =  (x > xmin ) & (x < xmax ) 
   return y / ( np.sum(y[xrange]) * (x[1]-x[0]) ) * normfactor


# Writting output files with PICCA format
def writeDelta(path_out, QSOloc, spectra, cat_type):
# save output
   out = fitsio.FITS(path_out+"/delta.fits.gz",'rw',clobber=True)
   print('Writting data from '+ str(len(QSOloc))+' QSOs')
   if ( cat_type == 'eBoss'):
      for i in range(0, len(QSOloc) ):
         hd = [ {'name':'RA','value':QSOloc[ i ][2],'comment':'Right Ascension [rad]'},
                 {'name':'DEC','value':QSOloc[ i ][1],'comment':'Declination [rad]'},
                 {'name':'Z','value':QSOloc[ i ][0],'comment':'Redshift'},
                 #{'name':'PMF','value':'{}-{}-{}'.format(d.plate,d.mjd,d.fid)},
                 {'name':'FIBER_ID','value':QSOloc[ i ][3],'comment':'Object identification'},
                 #{'name':'PLATE','value':d.plate},
                 #{'name':'MJD','value':d.mjd,'comment':'Modified Julian date'},
                 #{'name':'FIBERID','value':d.fid},
                 {'name':'ORDER','value':1,'comment':'Order of the continuum fit'},
         ]
         cols=[np.log10( spectra[i][0] ), spectra[i][1]*0, spectra[i][2], spectra[i][1]]
         names=['LOGLAM','DELTA','WEIGHT','CONT']
         units=['Log Angstrom','','','']
         comments = ['Log Lambda','Delta field','Pixel weights','Continuum']

         out.write(cols,names=names,header=hd,comment=comments,units=units,extname=str(QSOloc[ i ][3]))
   
   elif ( cat_type == 'Desi'):
      for i in range(0, len(QSOloc) ):
         hd = [ {'name':'RA','value':QSOloc[ i ][2],'comment':'Right Ascension [rad]'},
                 {'name':'DEC','value':QSOloc[ i ][1],'comment':'Declination [rad]'},
                 {'name':'Z','value':QSOloc[ i ][0],'comment':'Redshift'},
                 #{'name':'PMF','value':'{}-{}-{}'.format(d.plate,d.mjd,d.fid)},
                 {'name':'TARGET_ID','value':QSOloc[ i ][3],'comment':'Object identification'},
                 #{'name':'PLATE','value':d.plate},
                 #{'name':'MJD','value':d.mjd,'comment':'Modified Julian date'},
                 #{'name':'FIBERID','value':d.fid},
                 {'name':'ORDER','value':1,'comment':'Order of the continuum fit'},
         ]
         cols=[np.log10( spectra[i][0] ), spectra[i][1]*0, spectra[i][2], spectra[i][1]]
         names=['LOGLAM','DELTA','WEIGHT','CONT']
         units=['Log Angstrom','','','']
         comments = ['Log Lambda','Delta field','Pixel weights','Continuum']

         out.write(cols,names=names,header=hd,comment=comments,units=units,extname=str(QSOloc[ i ][3]))
   
   out.close()
   print('Done writting to file.')
   
# eBoss loading function, returns the same output as load_Desi
def load_eBoss(path_drq, path_spec, zmin, zmax, lmin, lmax):
   # eBoss Catalog load
   catalog = Table.read(path_drq)
   
   w = (catalog['THING_ID']>0) & (catalog['Z'] > zmin ) & (catalog['Z']< zmax ) & (catalog['RA']!=catalog['DEC'])& (catalog['RA']>0) & (catalog['DEC']>0)
   reduced_cat = catalog[w]
   reduced_cat = reduced_cat.group_by('PLATE')

   # thing_id = reduced_cat['THING_ID']
   fiberid = reduced_cat['FIBERID']
   plate = reduced_cat['PLATE']
   zqso = reduced_cat['Z']
   DEC = reduced_cat['DEC']
   RA = reduced_cat['RA']

   plate_list=[]
   for p,m in zip(reduced_cat['PLATE'],reduced_cat['MJD']):
        plate_list.append(str(p)+'/spPlate-'+str(p)+'-'+str(m)+'.fits')
   plate_list=np.unique(plate_list)

    #thisplate=plate_list[0].split("/")[1]  # Location
    #thisplate=plate_list[0].split("/")[0]  # Plate Number
   print('Found '+ str(  np.sum(w) ) + ' QSO spec. in ' +str(len(plate_list))+ ' files of catalog: '+ path_drq )
    
   spectra = []
   QSOloc = []
   ## begin for test
   for nplate in range ( 0, len(plate_list) ):      # len(plate_list)
      plate1=pyfits.open( path_spec+'/'+plate_list[nplate].split("/")[1] )
      thisplate=plate_list[nplate].split("/")[0]  # Plate Number

      wp = plate == int( thisplate )
      ids_=fiberid[wp]
      zqso_=zqso[wp]
      DEC_ = DEC[wp]
      RA_ = RA[wp]
      
      nqsoPlate_= ids_.shape[0]
      print( str(nplate) + ': Loading '+ str(nqsoPlate_) +' QSO spec. from plate: '+ plate_list[nplate].split("/")[1] )
      # Reading data from plate
      plugmap = plate1['PLUGMAP'].data
      # Searching for fiber of qso in data
      wp = np.in1d(plugmap['FIBERID'],ids_)
      # Applying mask to select only QSO
      small_plugmap = plugmap[wp]

      #Get the spectra
      flux=plate1[0].data
      #Get the weight
      ivar=plate1[1].data

      #Get the wavelenght
      pltheader=plate1[0].header
      coeff0=pltheader['COEFF0']
      coeff1=pltheader['COEFF1']
      logwave=coeff0+coeff1*np.arange(flux.shape[1])
      for i in range(0,nqsoPlate_):
            w_ = (10**logwave)/(1+zqso_[i])
            w_crop = ( w_ >= lmin ) & ( w_ <= lmax )
            w_ = w_[w_crop]
            flx = flux[ids_[i]-1][w_crop]
            ivr = ivar[ids_[i]-1][w_crop]
            flx = normalizeSpec( w_, flx, 1300, 1500)
            
            QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i], ids_[i]  )) )
            s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
            spectra.append( s )
   print('Reading done.')   
   return QSOloc, spectra

# Desi loading function, returns the same output as load_eBoss
def load_Desi(path_zcat, path_spec, zmin, zmax, lmin, lmax):
   # Desi Catalog load, beginning of function
   catalog = Table.read(path_zcat)
   qso_string = catalog['SPECTYPE'][0]
   
   w = (catalog['SPECTYPE']==qso_string ) & (catalog['Z'] > zmin ) & (catalog['Z']< zmax ) & (catalog['RA']!=catalog['DEC'])& (catalog['RA']>0) & (catalog['DEC']>0)
   reduced_cat = catalog[w]

   nest = True
   in_nside = 16

   targetid = reduced_cat['TARGETID']
   zqso = reduced_cat['Z']
   DEC = reduced_cat['DEC'] * np.pi/180
   RA = reduced_cat['RA'] * np.pi/180

   heal_pix = healpy.ang2pix(in_nside, sp.pi/2.-DEC, RA, nest)
   plate_list = np.unique(heal_pix)
   fi = glob.glob(path_spec+'/*/*/spectra*.fits*')
   print('Found', len(fi), 'spectra files.\n')
   fi_fix = []
   for i in range( 0, len(fi)):
       fi_fix.append( splitID(fi[i]) )

   fi_fix =  np.array( list(map(int, fi_fix)) ) 
   
   print('Found '+ str(  np.sum(w) ) + ' QSO spec. in ' + str( len(plate_list) ) + ' files.' )
   
   spectra = []
   QSOloc = []
   
   ## begin for test
   for nplate in range( 0, len(plate_list) ):      # len(plate_list)
      thisplate = plate_list[ nplate ]
      wp = heal_pix == int( thisplate )
      
      ids_ = targetid[wp]
      zqso_= zqso[wp]
      DEC_ = DEC[wp]
      RA_ = RA[wp]
      heal_pix_ = heal_pix[wp]
      
      # heal_pix: From healpy.ang2pix
      # plate_list: Unique from healpy.ang2pix \n
      # fi_fix: Plate id from directory list
      # fi: All files from glob
      nqsoPlate_= ids_.shape[0]
      # print( str(nplate) + ': Loading '+ str(nqsoPlate_) +' QSO spec. from file: '+ [] )      
      wpf = fi_fix == thisplate
      index = wpf * np.arange( len(fi_fix) )
      index = np.squeeze( index[wpf] )
      
      
      #print( thisplate,  nqsoPlate_ , len(fi_fix) )
      
      # print( thisplate, fi_fix[index], nqsoPlate_, fi[index] )
      print( str(nplate) + ': Loading '+ str(nqsoPlate_) +' QSO spec. from file: '+  str(thisplate) )
      
      spectra_base = desispec.io.read_spectra( fi[index] )   
      
      joint1 = np.in1d( spectra_base.wave['b'], spectra_base.wave['r'])
      joint2 = np.in1d( spectra_base.wave['r'], spectra_base.wave['b'])
            
      ll = np.concatenate( ( spectra_base.wave['b'][np.invert(joint1)] , spectra_base.wave['r'] ) )
      
      for i in range(0,nqsoPlate_):
         
         index = np.where( np.array( spectra_base.fibermap['TARGETID'].data ) == ids_.data[i] )
         index = np.squeeze(index)
         
         w_ = (ll)/(1+zqso_[ i ])
         w_crop = ( w_ >= lmin ) & ( w_ <= lmax )
         w_ = w_[w_crop] 
         
         intersec = ( spectra_base.ivar['b'][ index ][joint1]*spectra_base.flux['b'][ index ][joint1] + spectra_base.ivar['r'][ index ][joint2]*spectra_base.flux['r'][ index ][joint2] )
         intersec = intersec /( spectra_base.ivar['b'][ index ][joint1] + spectra_base.ivar['r'][ index ][joint2] )

         flx = np.concatenate( ( spectra_base.flux['b'][ index ][np.invert(joint1)], intersec, \
               spectra_base.flux['r'][ index ][np.invert(joint2)] ) )
         flx = flx[w_crop]
         
         intersec = ( spectra_base.ivar['b'][ index ][joint1] + spectra_base.ivar['r'][ index ][joint2] ) 
         ivr = np.concatenate( ( spectra_base.ivar['b'][ index ][np.invert(joint1)], intersec, \
               spectra_base.ivar['r'][ index ][np.invert(joint2)] ) )
         ivr = ivr[w_crop]
         
         flx = normalizeSpec( w_, flx, 1300, 1500)

         
         QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i], ids_[i] )) ) 
         s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
         spectra.append( s )
   
   print('Reading done.')
   return QSOloc, spectra



# EMPCA modeling function. 
def get_pca( spectra, niter, nvec):

   wwave = sp.arange( lmin, lmax, .1) 

   nbObj = len(spectra)
   # nbObj = 20

   pcaflux  = sp.zeros(( nbObj, wwave.size))
   pcaivar  = sp.zeros(( nbObj, wwave.size))

   for nspectra in range( 0, nbObj):
      pcaflux[nspectra], pcaivar[nspectra] = resample_flux( wwave, spectra[ nspectra ][0], spectra[ nspectra ][1], spectra[ nspectra ][2]) # interpolation

   pcaivar[pcaivar<0.] = 0.   # Remove if all measured bins are zero
   w = sp.sum(pcaivar,axis=0)>0.
   pcawave = wwave[w]
   pcaflux = pcaflux[:,w]
   pcaivar = pcaivar[:,w]
   ### Cap the ivar
   pcaivar[pcaivar>100.] = 100.

   ### Get the mean
   data_meanspec = sp.average(pcaflux,weights=pcaivar,axis=0) # Here, I get the mean spectrum.
   for i in range(nbObj):       #
      w = pcaivar[i]>0.        # subtracting the mean for each spectrum
      pcaflux[i,w] -= data_meanspec[w] #

   ### PCA
   print('INFO: Starting EMPCA')
   dmodel = empca.empca(pcaflux, weights=pcaivar, niter=niter, nvec=nvec)
        
   return dmodel, pcawave, pcaflux, pcaivar, data_meanspec


# Function for calculating the model continuuum using a exp. in 4 eig by default.
def get_continuum(wavelength,coeff,eigvec,mean_spec,n_vec=4,lmin=1300.0,lmax=1500.0,dw = 0.28):
   ### Choose the first four eigenvectors
   conti_mock = []
   for i in range(len(coeff)):
      spectram = []
      for j in range(n_vec):
         spectram.append(coeff[i][j]*eigvec[j])
      spectrasm = np.vstack(spectram)
      specm = np.sum(spectrasm,axis = 0)
      conti_mock += [specm]
   continuum_mock = np.vstack(conti_mock)
   ###
   new_wave = np.arange(600, 3000, dw)
   flux_mock = np.zeros((len(continuum_mock), new_wave.size))
   for i in range(len(continuum_mock)):
      flux_mock[i] = resample_flux(new_wave, wavelength, mean_spec+continuum_mock[i])
   ### Normalization
   integral_mock = []
   continuum_nor_mock = []
   for i in range(len(flux_mock)):
      sum2=0
      for j in range(len(flux_mock[i])):
         if lmin <= new_wave[j] < lmax :
            sum2+=(flux_mock[i][j])*(new_wave[j+1]-new_wave[j])
         elif (new_wave[j] > lmax):
            break
      integral_mock.append(sum2)
      continuum_nor_mock.append(8*flux_mock[i]/sum2)
   contin_mock = np.vstack(continuum_nor_mock)
   ### Normalization for the eigenvalues.
   coefficient = np.zeros((4,len(coeff)))
   for k in range(4):
      coefficient[k] = coeff[:,k]/integral_mock
   ### Stack
   stack_mock = np.mean(contin_mock,axis=0)     # mean continuum.
   std_stack_mock = np.std(contin_mock,axis=0)  # standard deviation.
   return new_wave, stack_mock, std_stack_mock, coefficient, contin_mock


# Function for printing the QSO continuum for both catalogs and its final covariaces
def plot_mean_cont(mwave,mflux,dwave,dflux,stdm,stdd,zmin,zmax,magmin,magmax,xmin,xmax,xminzoom,xmaxzoom):
   plt.figure(figsize=(14,15))
   plt.subplot(2,1,1)
   plt.title('Mean Continuum     {} < MAG < {} and {} < z < {}.'.format(magmin,magmax,zmin,zmax),fontsize = 20)
   plt.plot(dwave,dflux,'-', label='Stack DR14',alpha=0.7)
   #plt.fill_between(new_wave,stack_boss+std_stack_boss,stack_boss-std_stack_boss, label='std BOSS', color = 'y', alpha=0.5)
   plt.plot(mwave,mflux,'-', label='Stack MOCK',alpha=0.7)
   #plt.fill_between(new_wave,stack_mock+std_stack_mock,stack_mock-std_stack_mock, label='std MOCK', color = 'k', alpha=0.4)
   plt.xlim(xmin,xmax)
   #plt.ylim(-0.002,0.01)
   #axvline(940.93)
   plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
   plt.ylabel('$\mathrm{\overline{Normalized \enspace Flux}}$', fontsize = 20)
   plt.legend(fontsize='xx-large')
   plt.grid()

   mask = (dwave>900) & (dwave<1216)

   plt.subplot(2,1,2)
   plt.plot(dwave[mask],dflux[mask], label='Stack DR14')
   plt.fill_between(dwave[mask],dflux[mask]+stdd[mask],dflux[mask]-stdd[mask], label='std BOSS', color = 'y', alpha=0.5)
   plt.plot(mwave[mask],mflux[mask], label='Stack MOCK')
   plt.fill_between(mwave[mask],mflux[mask]+stdm[mask],mflux[mask]-stdm[mask], label='std MOCK', color = 'k', alpha=0.4)
   plt.xlim(xminzoom,xmaxzoom)
   plt.ylim(0,0.04)
   plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
   plt.ylabel('$\mathrm{\overline{Normalized \enspace Flux}}$', fontsize = 20)
   plt.legend(fontsize='xx-large')
   plt.grid()









############# Main Function

if __name__ == '__main__':
   
   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Overdensities calculation for Ly.a spectra')
   
   parser.add_argument('--path-cat',type=str,default=None,required=True,
        help='Catalog of objects in DRQ or Zcat format')
   
   parser.add_argument('--path-spec', type=str, default=None, required=True,
        help='Directory to spectra files')
   
   parser.add_argument('--path-out', type=str, default=None, required=True,
        help='Directory to output file(s)')

   parser.add_argument('--type', type=str, default=None, required=True,
        help='Catalog of objects in DRQ or Zcat format')


    #parser.add_argument('--log',type=str,default='input.log',required=False,
    #    help='Log input data')

   args = parser.parse_args()

   #####################################
   # Load QSO from plates (eBoss)
   ### Parameters
   #if 1:
   #   path_cat       = '/work/sfbeltranv/DR14_mini/DR14Q_v4_4m.fits'
   #   path_spec      = '/work3/desi_lya/data/eBOSS/dr15_all/spplates'
   #   cat_type       = 'eBoss'
   # for line correction (later)
   #path_lines     = '/work3/desi_lya/data/eBOSS/dr12_all/dr16-line-sky-mask.txt'
   #else:
   # Load QSO from fibers (Desi)
   #   path_cat       = '/work/sfbeltranv/DR14_mini/zcat_m.fits' 
   #   path_spec      = '/work3/desi_lya/mocks_quick/london/v9.0.0_small/spectra-16'
   #   cat_type       = 'Desi'

   path_drq       = args.path_cat
   path_spec      = args.path_spec
   path_out       = args.path_out
   cat_type       = args.type

   zmin = 2.
   zmax = 4.

   lmin = 1040
   lmax = 2000

   # Catalog load
   if ( cat_type == 'eBoss'):
      print('Catalog type: '+cat_type)
      QSOloc, spectra = load_eBoss(path_drq, path_spec, zmin, zmax, lmin, lmax)
      #writeDelta(path_out, QSOloc, spectra, 'eBoss')
    
   elif ( cat_type == 'Desi'):
      print('Catalog type: '+cat_type)
      QSOloc, spectra = load_Desi(path_drq, path_spec, zmin, zmax, lmin, lmax)
      #writeDelta(path_out, QSOloc, spectra, 'Desi')

   else:
      print('Wrong catalog type: '+cat_type)

   print( 'Done, loaded '+ str( len(spectra)) +' QSO spec. from catalog.'  )







