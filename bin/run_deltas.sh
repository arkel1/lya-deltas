#!/bin/bash





CAT='/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0_Y1/v9.0.0/desiY1-0.0/zcat.fits' 
SPEC='/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0_Y1/v9.0.0/desiY1-0.0/spectra-16'
OUT='/global/homes/s/sfbeltr/respaldo/out_deltas/lyadeltas-desi-Y1/deltas'
TYPE='Desi'



python Lya_deltas_do.py \
	--path-cat $CAT \
	--path-spec $SPEC \
	--path-out $OUT \
	--cat-type $TYPE \
    --fit-method PCA \
    --nspec 150 \
    --nproc 8