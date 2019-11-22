import numpy as np
import scipy as sp
from astropy.io import fits as pyfits
from astropy.table import Table
#from desispec.interpolation import resample_flux
#from desitarget.targetmask import desi_mask
#import desispec.io
#import sys
import os
import matplotlib.pyplot as plt


def load_eBoss(path_drq, path_spec, zmin, zmax):
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
            w_crop = ( w_ >= 1040 ) & ( w_ <= 1215.67 )
            w_ = w_[w_crop]
            flx = flux[ids_[i]-1][w_crop]
            ivr = ivar[ids_[i]-1][w_crop]
            
            QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i] )) )
            s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
            spectra.append( s )
            
    return QSOloc, spectra


def load_Desi(path_drq, path_spec, zmin, zmax):
    return 'maw location', 'maw spectra'


#####################################
# Load QSO from plates
### Parameters
path_drq       = '/work/sfbeltranv/DR14_mini/DR14Q_v4_4m.fits'
path_spec      = '/work3/desi_lya/data/eBOSS/dr15_all/spplates'
cat_type       = 'eBoss'
# for line correction (later)
#path_lines     = '/work3/desi_lya/data/eBOSS/dr12_all/dr16-line-sky-mask.txt'

zmin = 2
zmax = 4

# Catalog load
if ( cat_type == 'eBoss'):
    print(cat_type)
    QSOloc, spectra = load_eBoss(path_drq, path_spec, zmin, zmax)
elif ( cat_type == 'Desi'):
    print(cat_type)
    QSOloc, spectra = load_Desi(path_drq, path_spec, zmin, zmax)
else:
    print('Wrong catalog type: '+cat_type)

print( 'Done, loaded '+ str( len(spectra)) +' QSO spec. from catalog.'  )

