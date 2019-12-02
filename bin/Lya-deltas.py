import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as scpo
import os

from astropy.io import fits as pyfits
from astropy.table import Table

# Desi catalog test
import desispec.io
import glob
from itertools import compress
import healpy


def chi2( alpha, *args ):
   a,b = alpha
   w,flux,ivar = args
   return np.sum( (( flux - (a*w+b) )**2 ) * ivar )

def splitID(fi_str):
   fi_str= fi_str.split("spectra-16-")[1]
   return fi_str.split(".fits")[0]

def load_eBoss(path_drq, path_spec, zmin, zmax):
   # eBoss Catalog load
   catalog = Table.read(path_drq)
   
   lya = 1200 # 1215.67 1200
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
   print('Found '+ str(  np.sum(w) ) + ' QSO spec. in catalog: '+ path_drq )
    
   spectra = []
   QSOloc = []
   ## begin for test
   for nplate in range ( 0, len(plate_list) ):
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
            w_crop = ( w_ >= 1040 ) & ( w_ <= lya )
            w_ = w_[w_crop]
            flx = flux[ids_[i]-1][w_crop]
            ivr = ivar[ids_[i]-1][w_crop]
            
            QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i] )) )
            s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
            spectra.append( s )
   print('Reading done.')         
   return QSOloc, spectra


def load_Desi(path_zcat, path_spec, zmin, zmax):
   # Desi Catalog load, beginning of function
   catalog = Table.read(path_zcat)
   qso_string = catalog['SPECTYPE'][0]
   
   lya = 1200 # 1215.67 1200

   
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
        # print( str(i)+' of '+str(nqsoPlate_) )
         w_ = (ll)/(1+zqso_[ i ])
         w_crop = ( w_ >= 1040 ) & ( w_ <= lya )
         w_ = w_[w_crop] 
         
         intersec = ( spectra_base.ivar['b'][ i ][joint1]*spectra_base.flux['b'][ i ][joint1] + spectra_base.ivar['r'][ i ][joint2]*spectra_base.flux['r'][ i ][joint2] )
         intersec = intersec /( spectra_base.ivar['b'][ i ][joint1] + spectra_base.ivar['r'][ i ][joint2] )

         flx = np.concatenate( ( spectra_base.flux['b'][ i ][np.invert(joint1)], intersec, \
               spectra_base.flux['r'][ i ][np.invert(joint2)] ) )
         flx = flx[w_crop]
         
         intersec = ( spectra_base.ivar['b'][ i ][joint1] + spectra_base.ivar['r'][ i ][joint2] ) 
         ivr = np.concatenate( ( spectra_base.ivar['b'][ i ][np.invert(joint1)], intersec, \
               spectra_base.ivar['r'][ i ][np.invert(joint2)] ) )
         ivr = ivr[w_crop]
      
         QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i] )) ) 
         s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
         spectra.append( s )
   
   print('Reading done.')
   return QSOloc, spectra

############# Main Function

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Overdensities calculation for Ly.a spectra')

    parser.add_argument('--out-dir',type=str,default=None,required=True,
        help='Output directory')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ or Zcat format')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to spectra files')

    #parser.add_argument('--log',type=str,default='input.log',required=False,
    #    help='Log input data')

    parser.add_argument('--mode',type=str,default='eBoss',required=True,
        help='Open mode of the spectra files: eBoss or Desi')

    parser.add_argument('--zqso-min',type=float,default=None,required=False,
        help='Lower limit on quasar redshift from drq')

    parser.add_argument('--zqso-max',type=float,default=None,required=False,
        help='Upper limit on quasar redshift from drq')

    parser.add_argument('--keep-bal',action='store_true',required=False,
        help='Do not reject BALs in drq')

    parser.add_argument('--bi-max',type=float,required=False,default=None,
        help='Maximum CIV balnicity index in drq (overrides --keep-bal)')

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=5500.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min',type=float,default=1040.,required=False,
        help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',type=float,default=1200.,required=False,
        help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--rebin',type=int,default=3,required=False,
        help='Rebin wavelength grid by combining this number of adjacent pixels (ivar weight)')

    parser.add_argument('--npix-min',type=int,default=50,required=False,
        help='Minimum of rebined pixels')

    parser.add_argument('--dla-vac',type=str,default=None,required=False,
        help='DLA catalog file')

    parser.add_argument('--dla-mask',type=float,default=0.8,required=False,
        help='Lower limit on the DLA transmission. Transmissions below this number are masked')

    parser.add_argument('--absorber-vac',type=str,default=None,required=False,
        help='Absorber catalog file')

    parser.add_argument('--absorber-mask',type=float,default=2.5,required=False,
        help='Mask width on each side of the absorber central observed wavelength in units of 1e4*dlog10(lambda)')

    parser.add_argument('--mask-file',type=str,default=None,required=False,
        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--eta-min',type=float,default=0.5,required=False,
        help='Lower limit for eta')

    parser.add_argument('--eta-max',type=float,default=1.5,required=False,
        help='Upper limit for eta')

    parser.add_argument('--vlss-min',type=float,default=0.,required=False,
        help='Lower limit for variance LSS')

    parser.add_argument('--vlss-max',type=float,default=0.3,required=False,
        help='Upper limit for variance LSS')

    parser.add_argument('--delta-format',type=str,default=None,required=False,
        help='Format for Pk 1D: Pk1D')

    parser.add_argument('--use-ivar-as-weight', action='store_true', default=False,
        help='Use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0)')

    parser.add_argument('--use-constant-weight', action='store_true', default=False,
        help='Set all the delta weights to one (implemented as eta = 0, sigma_lss = 1, fudge = 0)')

    parser.add_argument('--order',type=int,default=1,required=False,
        help='Order of the log(lambda) polynomial for the continuum fit, by default 1.')

    parser.add_argument('--nit',type=int,default=5,required=False,
        help='Number of iterations to determine the mean continuum shape, LSS variances, etc.')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


    parser.add_argument('--use-mock-continuum', action='store_true', default = False,
            help='use the mock continuum for computing the deltas')

    args = parser.parse_args()







