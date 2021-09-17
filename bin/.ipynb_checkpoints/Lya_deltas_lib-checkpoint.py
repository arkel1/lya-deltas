import numpy as np
import scipy as sp
from astropy.io import fits as pyfits
from astropy.table import Table
import desispec.io
from desispec.interpolation import resample_flux

import os
from multiprocessing import Pool

import scipy.optimize as scpo
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import iminuit

import fitsio

# Desi catalog test
import glob
from itertools import compress
import healpy

import empca
import time


class DELTA():
   def __init__(self, w, delta, ivar, cont ):
      self.w = w   # must be in obs frame
      self.delta = delta
      self.ivar = ivar     # ivar weight with var correction
      self.cont = cont

class QSO:
   @classmethod
   def get_mcont(cls, w):
      raise NotImplementedError("Function should be specified at run-time")
   @classmethod
   def get_eta(cls, w):
      raise NotImplementedError("Function should be specified at run-time")      
   @classmethod
   def get_sigma(cls, w):
      raise NotImplementedError("Function should be specified at run-time")
   @classmethod
   def get_epsilon(cls, w):
      raise NotImplementedError("Function should be specified at run-time")
   
   def __init__(self, w, flux, ivar, z, ra, dec, thingid, fiberid, mjd):
      self.w = w    # rest frame
      self.flux = flux
      self.ivar = ivar    # for pipeline ivar
      self.z = z
      self.ra = ra
      self.dec = dec
      self.thingid = thingid
      self.fiberid = fiberid
      self.mjd = mjd
      self.delta = DELTA( w *( 1 + z ), flux, ivar, flux)

# def. of comovil distance calculation
def Rcomv(lam_):
   z = ( lam_ )/1216 - 1
   return np.interp(z, z_, r_)

# def. conversion of r, declination and ra to x y z
def coordC(r,ra,dec):
   return r*np.cos(dec)*np.cos(ra), r*np.cos(dec)*np.sin(ra), r*np.sin(dec)

# Normalize spectra with a scale factor to get spectra amplitude ~ 1
def normalizeSpec( x, y, xmin, xmax):
   normfactor = xmax-xmin
   xrange =  (x > xmin ) & (x < xmax ) 
   
   return y / ( np.sum(y[xrange]) * (x[1]-x[0]) ) * normfactor

# To crop the DESI spectra filenames for healpy
def splitID(fi_str):
   fi_str= fi_str.split("spectra-16-")[1]
   return fi_str.split(".fits")[0]


def load_pix( file ):
   
   global targetid
   global zqso
   global DEC
   global RA
   global heal_pix
   global lminrest
   global lmaxrest

   thisplate = splitID(file)
   
   wp = heal_pix == int( thisplate )

   ids_ = targetid[wp]
   zqso_= zqso[wp]
   DEC_ = DEC[wp]
   RA_ = RA[wp]
   heal_pix_ = heal_pix[wp]
   
   nqsoPlate_= ids_.shape[0]
      
   spectra_base = desispec.io.read_spectra( file )   
   
   joint1 = np.in1d( spectra_base.wave['b'], spectra_base.wave['r'])
   joint2 = np.in1d( spectra_base.wave['r'], spectra_base.wave['b'])
   
   ll = np.concatenate( ( spectra_base.wave['b'][np.invert(joint1)] , spectra_base.wave['r'] ) )
   
   data = []
      
   for i in range( 0, nqsoPlate_):   #nqsoPlate_

      index = np.where( np.array( spectra_base.fibermap['TARGETID'].data ) == ids_.data[i] )
      index = np.squeeze(index)

      w_ = (ll)/(1+zqso_[ i ])
      w_crop = ( w_ >= lminrest ) & ( w_ <= lmaxrest )
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
      #flx = normalizeSpec( w_, flx, 1300, 1500)

      w_crop = ( w_ >= 1040 ) & ( w_ <= 1500 ) & (ivr> 0 )
      w_ = w_[w_crop]
      flx = flx[w_crop]
      ivr = ivr[w_crop]

      data.append( QSO( w_, flx, ivr, zqso_[i], RA_[i],  DEC_[i], ids_[i], ids_[i], 1 ) )
   
   #print( file, ': ', ids_.shape[0], len(data) )
   return data

# Desi loading function, returns the same output as load_eBoss
def load_Desi_parallel(path_zcat, path_spec, zmin, zmax, lmin, lmax, multiC = 2, nspec = -1):
   # Desi Catalog load, beginning of function
   catalog = Table.read(path_zcat)
   qso_string = catalog['SPECTYPE'][0]

   w = (catalog['SPECTYPE']==qso_string ) & (catalog['Z'] >= zmin ) & (catalog['Z'] <= zmax ) 
   reduced_cat = catalog[w]

   if ( nspec > 1 and nspec < len(reduced_cat) ):
      print("Reducing catalog to "+str(nspec)+" spectra.")
      w = np.random.choice( len(reduced_cat), nspec, replace=False)
      reduced_cat = reduced_cat[w]
    
   nest = True
   in_nside = 16
   
   global targetid
   global zqso
   global DEC
   global RA
   global heal_pix
   global lminrest
   global lmaxrest
   
   lminrest = lmin
   lmaxrest = lmax
   
   targetid = reduced_cat['TARGETID']
   zqso = reduced_cat['Z']
   DEC = reduced_cat['DEC'] * np.pi/180
   RA = reduced_cat['RA'] * np.pi/180
   #LATE = reduced_cat['TILEID']
   
   heal_pix = healpy.ang2pix(in_nside, sp.pi/2.-DEC, RA, nest)
   plate_list = np.unique(heal_pix)
   
   #print('healpix len: ', len(plate_list), 'healpix values: ',  plate_list )
   fi = glob.glob(path_spec+'/*/*/spectra*.fits*')
   print('Found', len(fi), 'spectra files.\n')
   
   fi_fix = []
   for i in range( 0, len(fi) ):
       fi_fix.append( splitID(fi[i]) )

   fi_fix =  np.array( list(map(int, fi_fix)) ) 
   
   print('Found '+ str(  np.sum(w) ) + ' QSO spec. in ' + str( len(plate_list) ) + ' files.' )

                            # from healpix.  #found files
   wpf_ = np.where( np.in1d(   plate_list,   fi_fix) )[0]
        # to crop catalog files to N: wpf_[0:N]
   
   fi_load = [ fi[index] for index in wpf_ ] 

   pool = Pool( processes = multiC )   
   data_pix = pool.map( load_pix , fi_load)
   pool.close()
   
   QSOlist = []

   for i in range( len (data_pix ) ):
      QSOlist.extend( data_pix[i] )
   
   del data_pix
   return QSOlist

# eBoss loading function, returns the same output as load_Desi
def load_eBoss(path_drq, path_spec, zmin, zmax, lmin, lmax):
   # eBoss Catalog load
   catalog = Table.read(path_drq)
   w = (catalog['THING_ID']>0) & (catalog['Z'] > zmin ) & (catalog['Z']< zmax ) & (catalog['RA']!=catalog['DEC'])& (catalog['RA']>0) & (catalog['DEC']>0)
   reduced_cat = catalog[w]
   reduced_cat = reduced_cat.group_by('PLATE')
   nest = True
   in_nside = 16
   thing_id = reduced_cat['THING_ID']
   fiberid = reduced_cat['FIBERID']
   plate = reduced_cat['PLATE']
   zqso = reduced_cat['Z']
   DEC = reduced_cat['DEC']* np.pi/180
   RA = reduced_cat['RA']* np.pi/180
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
      tids_=thing_id[wp]
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
            
            heal_pix = healpy.ang2pix(in_nside, sp.pi/2.-DEC_[i], RA_[i], nest)
            
            QSOloc.append( np.hstack(( zqso_[i], DEC_[i], RA_[i], tids_[i]  )) )
            s=np.vstack( ( w_.conj().transpose(), flx.conj().transpose(), ivr.conj().transpose() ) )
            spectra.append( s )
   print('Reading done.')   
   return QSOloc, spectra

# Writting output files with PICCA format
def writeDelta( path_out, QSOlist, nside_ = 8, nest_ = False ):
# save output
   ra  = np.array( [ QSOlist[i].ra  for i in range( len(QSOlist) ) ]  )
   dec = np.array( [ QSOlist[i].dec for i in range( len(QSOlist) ) ]  )
   pix = healpy.ang2pix( nside_ , sp.pi/2.-dec, ra, nest=nest_ )
   tags = np.unique(pix)
   print('Writting data from '+ str(len(QSOlist))+' QSOs.')
   for j in range(0, len(tags) ):   #  len(tags)
      out = fitsio.FITS(path_out+'/delta-'+ str(tags[j]) +'.fits.gz','rw',clobber=True)
      mask = pix == int( tags[j])
      are  =  np.squeeze( np.where(mask) )
        
      try:
         iter(are)
      except TypeError:
         are = [are]
      
      for i in are:
         hd = [ {'name':'RA','value':QSOlist[ i ].ra,'comment':'Right Ascension [rad]'},
                 {'name':'DEC','value':QSOlist[ i ].dec,'comment':'Declination [rad]'},
                 {'name':'Z','value':QSOlist[ i ].z,'comment':'Redshift'},
                 {'name':'PMF','value':'{}-{}-{}'.format(1,1,1)},
                 {'name':'THING_ID','value':QSOlist[ i ].thingid,'comment':'Object identification'},
                 {'name':'PLATE','value':tags[j]},
                 {'name':'MJD','value':1,'comment':'Modified Julian date'},
                 {'name':'FIBERID','value':QSOlist[ i ].thingid},
                 {'name':'ORDER','value':1,'comment':'Order of the continuum fit'},
         ]
         cols=[np.log10( QSOlist[i].delta.w ), QSOlist[i].delta.delta, QSOlist[i].delta.ivar, QSOlist[i].delta.cont    ]    
         names=['LOGLAM','DELTA','WEIGHT','CONT']
         units=['Log Angstrom','','','']
         comments = ['Log Lambda','Delta field','Pixel weights','Continuum']
         out.write(cols,names=names,header=hd,comment=comments,units=units,extname=str(QSOlist[ i ].thingid) )

      out.close()
      print('Done writting to file '+str(tags[j])+'.' )
   print('All files written.')


def load_piccaDelta(file):
   catalog = pyfits.open( file ) 
   QSOlist = []
   for j in range(1, len(catalog) ):
      qso = QSO( 10**catalog[j].data['LOGLAM'] / (1 + catalog[j].header['Z'] ), (catalog[j].data['DELTA']+1)*catalog[j].data['CONT'], catalog[j].data['WEIGHT'], catalog[j].header['Z'], catalog[j].header['RA'],  catalog[j].header['DEC'], catalog[j].header['THING_ID'], catalog[j].header['THING_ID'], 1 )     
      qso.delta.cont = catalog[j].data['CONT']
      QSOlist.append( qso )

   return QSOlist


def load_Picca(dir, multiC = 2):
   ################# delta reading and loading with thing_id
   '''
   dir  = '/work/sfbeltranv/output/deltas'
   catalog = pyfits.open(dir + 'delta-100.fits.gz')
   catalog[1].header
   plt.plot( catalog[1].data['LOGLAM'], catalog[1].data['DELTA'] )
   catalog[1].header
   catalog[1].header['THING_ID']
   '''
   ################## script to load al delta-healpix piles

   fi = glob.glob(dir+'/*')
   print( 'Loading from ', len(fi), ' delta files.' )
   QSOlist = []
   pool = Pool( processes = multiC )
   data_pix = pool.map( load_piccaDelta , fi)
   pool.close()
   for i in range( len (data_pix ) ):
      QSOlist.extend( data_pix[i] )
   del data_pix
   print('Loaded', len(QSOlist), 'QSO.')
   return QSOlist


def calcMeanCont(QSOlist, mean = 0, lmin = 1040, lmax=1200):
   dll = 1
   nstack = int((lmax - lmin)/dll)+1
   ll = lmin + ( np.arange(nstack)+0.5 ) * dll
   mcont = np.zeros(nstack)
   wcont = np.zeros(nstack)
   for i in range( len(QSOlist) ):   # len(deltas)
      wave = QSOlist[i].w
      bins = ( (wave - lmin) / dll + 0.5 ).astype(int)
      
      varpipe = 1 / ( QSOlist[i].ivar*QSOlist[i].delta.cont )
      var = QSOlist[i].get_eta(wave)*varpipe + QSOlist[i].get_sigma(wave) + QSOlist[i].get_epsilon(wave)/varpipe
      weight = 1 / var
      
      c = np.bincount( bins, weights = QSOlist[i].flux / QSOlist[i].delta.cont*weight )
      mcont[:len(c)]+=c 
      c = np.bincount( bins, weights = weight )
      wcont[:len(c)]+=c
   w = wcont > 0
   mcont[w] /= wcont[w]
   mcont /= mcont.mean()
   wcont /= wcont[w].mean()
   #plt.plot(ll,mcont)
   #plt.plot(ll,wcont)
   #plt.show()
   return ll, mcont, wcont
  
# def. Function to minimize the error of the linear 
# regression * mcont for the continuum fit 
def chi2cont( alpha, *args ):
   a,b = alpha
   w, flux, ivar, mcont, eta, sigma, epsilon = args
   newc = (a*w+b) * mcont
   varpipe = 1 / ivar / newc**2
   variance = eta*varpipe + sigma + epsilon/varpipe
   weight = 1 / newc**2 / variance
   chi2_contribution = (flux - newc)**2 * weight
   return chi2_contribution.sum() - np.log(weight).sum()


def calcDelta( qso, lmin=1040, lmax=1200 ):
   w_o, ivar_, flux_ = [qso.w, qso.ivar, qso.flux]
   wm = ( ivar_ > 0 ) & ( w_o >= lmin ) & ( w_o <= lmax )
   w_ = w_o[wm] 
   ivar_ = ivar_[wm]  
   flux_ = flux_[wm]
   mcont_ = qso.get_mcont(w_)
   eta = qso.get_eta(w_)
   sigma = qso.get_sigma(w_)
   epsilon = qso.get_epsilon(w_)
   param = scpo.minimize(chi2cont, (0,1), args=( w_, flux_, ivar_, mcont_, eta, sigma, epsilon  ) ); #COBYLA
   CF = ( param.x[0]*w_+param.x[1] ) *  mcont_
   delta_ = flux_ / CF - 1 
   qso.delta.w = w_  * ( 1 + qso.z )
   qso.delta.delta = delta_ #- np.sum(delta_) / len(delta_)     # zero Centered delta
   qso.delta.cont = CF
   
   var_pipe = 1. / ivar_ / CF**2
   variance = eta * var_pipe + sigma + epsilon / var_pipe
   weights = 1.0 / CF**2 / variance
   qso.delta.ivar = weights
   return qso

def recalcDeltas(QSOlist, multiC = 2):
   pool = Pool( processes = multiC )   
   QSOlist = pool.map( calcDelta, QSOlist )
   pool.close()
   return QSOlist


def minimizeVariances(QSOlist, lmin=3600, lmax=5800, lrmin=1040, lrmax=1200, num_bins = 20, limit_eta=(0.5, 1.5), limit_var_lss=(0., 0.3)):       
   # Bins for final correction saved in bins with groups of wavelenghts
   get_mcont = QSO.get_mcont
   
   eta = np.zeros(num_bins)
   var_lss = np.zeros(num_bins)
   epsilon = np.zeros(num_bins)
   error_eta = np.zeros(num_bins)
   error_var_lss = np.zeros(num_bins)
   error_epsilon = np.zeros(num_bins)
   num_pixels = np.zeros(num_bins)
   ll = (lmin + (np.arange(num_bins) + .5) *
                  (lmax - lmin) / num_bins)
   # Value list for pipeline correction
   num_var_bins = 100
   var_pipe_min = np.log10(1e-5)
   var_pipe_max = np.log10(20)
   var_pipe_values = 10**(var_pipe_min +
                           ((np.arange(num_var_bins) + .5) *
                            (var_pipe_max - var_pipe_min) / num_var_bins))
   
   var_delta = np.zeros(num_bins * num_var_bins)
   mean_delta = np.zeros(num_bins * num_var_bins)
   var2_delta = np.zeros(num_bins * num_var_bins)
   count = np.zeros(num_bins * num_var_bins)
   num_qso = np.zeros(num_bins * num_var_bins)
   print('Binning deltas')
   for i in range( len(QSOlist) ):   # len(deltas)
      # print('delta '+str(i)+' of '+str( len(deltas) )+'.' )
      # wave, delta, ivar, flx, CF
      wave = QSOlist[i].delta.w
      
      var_pipe =  1 / ( QSOlist[i].ivar * (QSOlist[i].delta.cont)**2 )
      w = ((np.log10(var_pipe) > var_pipe_min) &
                 (np.log10(var_pipe) < var_pipe_max))
      log_lambda_bins = ((wave  - lmin) /
                               (lmax - lmin) * num_bins).astype(int)
      var_pipe_bins = np.floor(
                (np.log10(var_pipe) - var_pipe_min) /
                (var_pipe_max - var_pipe_min) * num_var_bins).astype(int)
      log_lambda_bins = log_lambda_bins[w]
      var_pipe_bins = var_pipe_bins[w]
      bins = var_pipe_bins + num_var_bins * log_lambda_bins
      # compute deltas
      delta = QSOlist[i].delta.delta
      delta = delta[w]
      
      # add contributions to delta statistics
      rebin = np.bincount(bins, weights=delta)
      mean_delta[:len(rebin)] += rebin

      rebin = np.bincount(bins, weights=delta**2)
      var_delta[:len(rebin)] += rebin

      rebin = np.bincount(bins, weights=delta**4)
      var2_delta[:len(rebin)] += rebin

      rebin = np.bincount(bins)
      count[:len(rebin)] += rebin
      num_qso[np.unique(bins)] += 1
   
   w = count > 0
   var_delta[w] /= count[w]
   mean_delta[w] /= count[w]
   var_delta -= mean_delta**2
   var2_delta[w] /= count[w]
   var2_delta -= var_delta**2
   var2_delta[w] /= count[w]
   # fit the functions eta, var_lss, and fudge
   chi2_in_bin = np.zeros(num_bins)
   epsilon_ref = 1e-7
   print('Deltas calculated, minimizing var bins in lambda')
   for index in range(num_bins):
      #print('Bin '+str(index)+' of '+str(num_bins)+'.' )
      # pylint: disable-msg=cell-var-from-loop
      # this function is defined differntly at each step of the loop
      def chi2varfit(eta, var_lss, epsilon):
         variance = eta * var_pipe_values + var_lss + epsilon*epsilon_ref / var_pipe_values
         chi2_contribution = (var_delta[index * num_var_bins:(index + 1) * num_var_bins] - variance)
         weights = var2_delta[index * num_var_bins:(index + 1) * num_var_bins]
         w = num_qso[index * num_var_bins:(index + 1) * num_var_bins] > 100
         return np.sum(chi2_contribution[w]**2 / weights[w])
      minimizer = iminuit.Minuit(chi2varfit,
                       name=("eta", "var_lss", "epsilon"),
                       eta=1.,
                       var_lss=0.1,
                       epsilon=1.,
                       error_eta=0.05,
                       error_var_lss=0.05,
                       error_epsilon=0.05,
                       errordef=1.,
                       print_level=0,
                       limit_eta=limit_eta,
                       limit_var_lss=limit_var_lss,
                       limit_epsilon=(0, None))
      minimizer.migrad()
      if minimizer.migrad_ok():
         minimizer.hesse()
         eta[index] = minimizer.values["eta"]
         var_lss[index] = minimizer.values["var_lss"]
         epsilon[index] = minimizer.values["epsilon"] * epsilon_ref
         error_eta[index] = minimizer.errors["eta"]
         error_var_lss[index] = minimizer.errors["var_lss"]
         error_epsilon[index] = minimizer.errors["epsilon"] * epsilon_ref
      else:
         eta[index] = 1.
         var_lss[index] = 0.1
         epsilon[index] = 1. * epsilon_ref
         error_eta[index] = 0.
         error_var_lss[index] = 0.
         error_epsilon[index] = 0.
      num_pixels[index] = count[index * num_var_bins:(index + 1) *
                         num_var_bins].sum()
      chi2_in_bin[index] = minimizer.fval
   print('Done iterating')   
   return (ll, eta, var_lss, epsilon, num_pixels, var_pipe_values,
            var_delta.reshape(num_bins, -1), var2_delta.reshape(num_bins, -1),
            count.reshape(num_bins, -1), num_qso.reshape(num_bins, -1),
            chi2_in_bin, error_eta, error_var_lss, error_epsilon)


# EMPCA modeling function. 
def get_PCA_model( QSOlist, niter, nvec, lmin = 1040, lmax = 1600 ):
   wwave = sp.arange( lmin, lmax, .1) 
   nbObj = len(QSOlist)
   # nbObj = 20
   pcaflux  = sp.zeros(( nbObj, wwave.size))
   pcaivar  = sp.zeros(( nbObj, wwave.size))

   for nspectra in range( 0, nbObj):
      pcaflux[nspectra], pcaivar[nspectra] = resample_flux( wwave, QSOlist[ nspectra ].w, QSOlist[ nspectra ].flux, QSOlist[ nspectra ].ivar) # interpolation

   pcaivar[pcaivar<0.] = 0.   # Remove if all measured bins are zero
   w = sp.sum(pcaivar,axis=0)>0.
   pcawave = wwave[w]
   pcaflux = pcaflux[:,w]
   pcaivar = pcaivar[:,w]
   ### Cap the ivar
   pcaivar[pcaivar>100.] = 100.

   ### Get the mean
   data_meanspec = sp.average(pcaflux,weights=pcaivar,axis=0)
   for i in range(nbObj):
      w = pcaivar[i]>0.        # subtracting the mean for each spectrum
      pcaflux[i,w] -= data_meanspec[w] #
   ### PCA
   print('Starting EMPCA: ')
   dmodel = empca.empca(pcaflux, weights=pcaivar, niter=niter, nvec=nvec)
   return dmodel, pcawave, pcaflux, pcaivar, data_meanspec

# Function for calculating the model continuuum using a expansion of 4 coef*eig by default.
def get_PCA_continuum( wavelength, coeff, eigvec, mean_spec, n_vec=4, lmin = 1300, lmax = 1500 ):
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
   flux_mock = np.zeros((len(continuum_mock), wavelength.size))
   for i in range(len(continuum_mock)):
      flux_mock[i] =  mean_spec+continuum_mock[i]
   ### Normalization
   integral_mock = []
   continuum_nor_mock = []
   for i in range(len(flux_mock)):
      sum2=0
      for j in range(len(flux_mock[i])):
         if lmin <= wavelength[j] < lmax :
            sum2+=(flux_mock[i][j])*(wavelength[j+1]-wavelength[j])
         elif (wavelength[j] > lmax):
            break
      integral_mock.append(sum2)
      continuum_nor_mock.append(flux_mock[i]/sum2)
   contin_mock = np.vstack(continuum_nor_mock)
  ### Normalization for the eigenvalues.
   coefficient = np.zeros((n_vec,len(coeff)))
   for k in range(n_vec):
      coefficient[k] = coeff[:,k] /integral_mock
   ### Stack
   stack_mock = np.mean(contin_mock,axis=0)     # mean continuum.
   std_stack_mock = np.std(contin_mock,axis=0)  # standard deviation.
   return wavelength, stack_mock, std_stack_mock, coefficient, contin_mock

  
### PCA delta calculation and centering
def get_PCA_deltas( QSOlist, pcawave, pcamcont, pcamcontstd, pcacontin_mock, lamin=1040, lamax=1200):
   deltas = []
   for i in range( len(QSOlist) ):         # len(QSOlist)
      dwave = QSOlist[ i ].w
      wmask = ( dwave >= lamin ) & ( dwave <= lamax )
      dwave = dwave[ wmask ]
      flux  = QSOlist[ i ].flux
      flux  = flux[ wmask]                        # orig flux
      ivr  =  QSOlist[ i ].ivar 
      ivr = ivr[ wmask]
      
      # pcawave, pcacont from cont. fitting with PCA to QSO delta ticks 
      # pcacont from pcawave to dwave
      cont  = resample_flux(dwave, pcawave, pcacontin_mock[i]  )   # continuum to dwave grid
                                                            # ivar=pcamcontstd
      delta = flux / cont - 1
      
      s=np.vstack( ( dwave.conj().transpose(), delta.conj().transpose(), ivr.conj().transpose(), cont.conj().transpose()  ) )
      QSOlist[ i ].delta.w = dwave * ( 1 + QSOlist[ i ].z )   # restframe to selframe
      QSOlist[ i ].delta.delta = delta - np.sum(delta) / len(delta)     # zero Centered delta
      QSOlist[ i ].delta.cont = cont
      QSOlist[ i ].delta.ivar = ivr
      
   return QSOlist



def blindQSO(qso):
   global lol_, l_, Zmz_, Z_, stack

   #print('\t\t QSO ', j, ' of ', len(catalog), '.' )
   # Z QSO ap shift
   Za = qso.z
   Z_rebin = np.interp( Za, Z_, Zmz_ )
   qso.z = Za + Z_rebin

   # QSO forest ap shift with interval conservation
   l = qso.w * (1 + Za)
   lol_rebin = resample_flux( l, l_, lol_ )

   flux = ( qso.flux  ) * qso.get_mcont( qso.w ) *  stack( l )
   
   #dx = (l[1]-l[0])
   #f = ( lol_rebin - 1 )
   #A = np.sum(f)*dx
   #lol_rebin = ( lol_rebin - 1 ) - A/(dx*len(f)) + 1

   l_rebin = lol_rebin*l
   #l_rebin = l_rebin - l_rebin[int(len(l_rebin)/2)] + l[int(len(l)/2)]
   
   llog = np.log10(l)
   #l2 = l-( l[0]-l_rebin[0] )
   l2 = 10**(  np.arange( np.log10( np.min(l_rebin) ), np.log10( np.max(l_rebin) ), llog[1]-llog[0] )    )
   
   flux, ivar = resample_flux( l2, l_rebin, flux, ivar=qso.ivar )
   delta, ivar2 = resample_flux( l2, l_rebin, qso.delta.delta, ivar=qso.delta.ivar )
   cont = resample_flux( l2, l_rebin, qso.delta.cont  )

   qso.w = l2 / (1 + qso.z)
   qso.delta.w = l2
   
   qso.flux =  flux 
   qso.ivar =  ivar
   
   qso.delta.delta = delta
   qso.delta.ivar = ivar2
   
   qso.delta.cont = cont

   return qso

def blindDeltas( QSOlist, multiC = 2):
   global lol_, l_, Zmz_, Z_
   
   lol_ = np.load('/work/sfbeltranv/lya-blinding/lol.npy')
   l_ = np.load('/work/sfbeltranv/lya-blinding/l.npy')
   
   Zmz_ = np.load('/work/sfbeltranv/lya-blinding/zmz.npy')
   Z_ = np.load('/work/sfbeltranv/lya-blinding/z.npy')
   
   pool = Pool( processes = multiC )   
   QSOlist = pool.map( blindQSO, QSOlist )
   pool.close()
   
   print("Deltas blinded.")
   return QSOlist