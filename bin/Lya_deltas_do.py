import sys
Lpath = "/work/sfbeltranv/lya-deltas/bin/"
sys.path.append(Lpath)

from Lya_deltas_lib import *

############# Main Function

if __name__ == '__main__':
   
   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Overdensities calculation for Ly.a spectra')
   
   parser.add_argument('--cat-type', type=str, default='Desi', required=False,
        help='Format of catalog of objects in DRQ (eBoss) or Zcat (Desi), Desi by default')    
    
   parser.add_argument('--path-cat', type=str, default=None, required=True,
        help='Path to catalog of objects in DRQ or Zcat format')
   
   parser.add_argument('--path-spec', type=str, default=None, required=True,
        help='Directory to spectra files')
   
   parser.add_argument('--path-out', type=str, default=None, required=True,
        help='Directory to output files')
   
   parser.add_argument('--fit-method', type=str, default='chi2', required=False,
        help='chi2 for fitting iteration or PCA')
   
   parser.add_argument('--chi2-use-ivar', type=bool, default=False, required=False,
        help='Use ivar as weight instead of (4) in arXiv:2007.08995v2')
   
   parser.add_argument('--chi2-iter', type=int, default=5, required=False,
        help='Number of processors for parallel using pool')
     
   parser.add_argument('--PCA-load', type=int, default=0, required=False,
        help='(0) calculate PCA from scratch, (1) load PCA from previous calculation \n\t or (2) calculate PCA and save')
    
   parser.add_argument('--nproc', type=int, default=2, required=False,
        help='Number of processors for parallel calc using pool')
   
   parser.add_argument('--nspec', type=int, default=None, required=False,
                        help='Maximum number of spectra to read')
   
   parser.add_argument('--zqso-min', type=float, default=2., required=False,
        help='Lower limit on quasar redshift')
   
   parser.add_argument('--zqso-max', type=float, default=4.288461538461538, required=False,
        help='Upper limit on quasar redshift')
   
   parser.add_argument('--l-min', type=float, default=1040, required=False,
        help='Lower limit on wavelenght')
   
   parser.add_argument('--l-max', type=float, default=2000, required=False,
        help='Upper limit on wavelenght')
  
   #parser.add_argument('--log',type=str,default='input.log',required=False,
   #     help='Log input data')
   
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
   #   cat_type       = 'Desi'1

   path_drq       = args.path_cat
   path_spec      = args.path_spec
   path_out       = args.path_out

   zmin = args.zqso_min
   zmax = args.zqso_max

   lmin = args.l_min
   lmax = args.l_max

   # Catalog load
   if ( args.cat_type == 'eBoss'):
       print(cat_type)
       QSOlist = load_eBoss(path_drq, path_spec, zmin, zmax, lmin, lmax)

   elif ( args.cat_type == 'Desi'):
       print(cat_type)
       QSOlist = load_Desi_parallel(path_drq, path_spec, zmin, zmax, lmin, lmax, multiC = args.nproc)

   else:
       print("Wrong catalog type: "+cat_type+". Use eBoss or Desi")
   
   print("Loaded "+ str( len(QSOlist)) +" QSO spec. from catalog.")
   print("Calculating Deltas:")

   if ( args.fit_method == 'PCA' ):
      if ( args.PCA_load != 1 ):     # Calculate cont with current data
         pcamodel, pcawave, pcaflux, pcaivar, data_meanspec = get_PCA_model(QSOlist, niter=10, nvec=10)
         pcawave, pcamcont, pcamcontstd, pcacoeff, pcacontin_mock = get_PCA_continuum( pcawave, pcamodel.coeff, pcamodel.eigvec, data_meanspec)
         if args.PCA_load == 2:
            np.save('wave.npy', pcawave)
            np.save('meanf.npy', pcamcont)
            np.save('meanstd.npy', pcamcontstd)         
      else:     # Load continuum from a 300K QSO catalog
         pcawave     = np.load('wave.npy')
         pcamcont    = np.load('meanf.npy')
         pcamcontstd = np.load('meanstd.npy')
         
      QSOlist = get_PCA_deltas(QSOlist, pcawave, pcamcont, pcamcontstd, pcacontin_mock ) 
   
   if ( args.fit_method == 'chi2' and args.chi2_use_ivar==True ):
      QSOlist = getdeltas_LinMinimize(QSOlist)
   
   if ( args.fit_method == 'chi2' and args.chi2_use_ivar==False ):
      # Temporal values to interpolate weights of obs frame values.
      lmin = 3000
      lmax = 6000
      ltemp = (lmin + np.arange(2) * (lmax - lmin))
      QSO.get_mcont = interp1d(ltemp, 1 + np.zeros(2), fill_value="extrapolate")
      QSO.get_eta = interp1d(ltemp, 1+ np.zeros(2), fill_value="extrapolate" )
      QSO.get_sigma = interp1d(ltemp, 1+ np.zeros(2), fill_value="extrapolate" )
      QSO.get_epsilon = interp1d(ltemp, 1+ np.zeros(2), fill_value="extrapolate" )
      
      for i in range( args.chi2_iter ):  # num of iterations to minimize weights
         print('Iteration '+str(i)+' of 5.')

         numbers = np.arange(0, len(QSOlist))
         QSOlist = recalcDeltas( QSOlist, multiC = args.nproc) 

         llcont, mcont, wcont = calcMeanCont(QSOlist) 
         print('Mean continuum calculated')
         newcont = QSO.get_mcont(llcont) * mcont
         QSO.get_mcont = interp1d(llcont, newcont, fill_value="extrapolate")

         if i == 0 : # Recalc deltas in first approx
            QSOlist = recalcDeltas( QSOlist, multiC = args.nproc) 

         (ll, eta, sigma, epsilon, num_pixels, var_pipe_values,
                          var_delta, var2_delta, count, num_qso, chi2_in_bin, error_eta,
                          error_var_lss, error_epsilon) = minimizeVariances(QSOlist)

         w = num_pixels > 0
         get_eta = interp1d(ll[w],
                                 eta[w],
                                 fill_value="extrapolate",
                                 kind="nearest")
         get_sigma = interp1d(ll[w],
                                 sigma[w],
                                 fill_value="extrapolate",
                                 kind="nearest")
         get_epsilon = interp1d(ll[w],
                                 epsilon[w],
                                 fill_value="extrapolate",
                                 kind="nearest")
		 
         # Update function to interp weights of class QSO	
         QSO.get_eta = get_eta
         QSO.get_sigma = get_sigma
         QSO.get_epsilon = get_epsilon
   
   # Recalculate deltas after final iteration
   QSOlist = recalcDeltas( QSOlist, multiC = args.nproc) 

   if not os.path.exists(path_out):
        os.makedirs(path_out)

   writeDelta( path_out, QSOlist, nside_ = 8, nest_ = False )



