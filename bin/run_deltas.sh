#!/bin/bash



if (( $1 > 1 )); then
   CAT='/work/sfbeltranv/DR14_mini/DR14Q_v4_4m.fits'
   SPEC='/work3/desi_lya/data/eBOSS/dr15_all/spplates'
   OUT='/work/sfbeltranv/output/lya-deltas'
   TYPE='eBoss'

else
   CAT='/work/sfbeltranv/DR14_mini/zcat_m.fits' 
   SPEC='/work3/desi_lya/mocks_quick/london/v9.0.0_small/spectra-16'
   OUT='/work/sfbeltranv/output/lya-deltas'
   TYPE='Desi'

fi


python Lya-deltas.py \
	--path-cat $CAT \
	--path-spec $SPEC \
    --path-out $OUT \
	--type $TYPE
