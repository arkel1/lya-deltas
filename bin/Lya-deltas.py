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


import argparse

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

   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Overdensities calculation for Ly.a spectra')

   parser.add_argument('--path-cat',type=str,default=None,required=True,
        help='Catalog of objects in DRQ or Zcat format')

   parser.add_argument('--path-spec', type=str, default=None, required=True,
        help='Directory to spectra files')

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

   path_cat       = args.path_cat
   path_spec      = args.path_spec
   cat_type       = args.type

   zmin = 2.
   zmax = 4.

   # Catalog load
   if ( cat_type == 'eBoss'):
      print(cat_type)
      QSOloc, spectra = load_eBoss(path_cat, path_spec, zmin, zmax)
   elif ( cat_type == 'Desi'):
      print(cat_type)
      QSOloc, spectra = load_Desi(path_cat, path_spec, zmin, zmax)
   else:
      print('Wrong catalog type: '+cat_type)

   print( 'Done, loaded '+ str( len(spectra)) +' QSO spec. from catalog.'  )







