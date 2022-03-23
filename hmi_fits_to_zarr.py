#!/usr/bin/python
#Given:
#   a folder --src containing fits files
#   a folder --target to contain zarr files
#   an integer --scale (a proper divisor of 512) containing the target output size
#Converts the source fits to target zarr files:
#   -Rescaling the sun to a constant pixel size and correcting for invalid interpolation
#   -Save all fits header information to meta data in zarr  

import os, pdb
import numpy as np
import sunpy.io
import glob
from tqdm import tqdm
import zarr
from sunpy.map import Map
import skimage.transform
import gcsfs
from numcodecs import Blosc, Delta
import warnings
import argparse

warnings.filterwarnings("ignore")

trgtAS = 976.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--target',dest='target',required=True)
    parser.add_argument('--scale',dest='scale',required=True,type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print(args)
    src, target, scale = args.src, args.target, args.scale

    if not os.path.exists(target):
        os.mkdir(target)

    divideFactor = np.int(512 / scale)

    filelist_bx  = sorted(glob.glob(src+'*bx.fits'))
    filelist_by = sorted(glob.glob(src+'*by.fits'))
    filelist_bz  = sorted(glob.glob(src+'*bz.fits'))

    store = zarr.DirectoryStore(target+'sdomlv2_hmi_2011.zarr')
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=store,overwrite=True)
    year = root.create_group('2011')
    bx = year.create_dataset('Bx',shape=(np.shape(filelist_bx)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    by = year.create_dataset('By',shape=(np.shape(filelist_by)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    bz = year.create_dataset('Bz',shape=(np.shape(filelist_bz)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)

    # Process BX
    Xd = Map(filelist_bx[-1])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            vars()[key] = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_bx)):
        Xd = Map(file)
        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple' and key != 'history':
                vars()[key].append(Xd.meta[key])
        pixlunit.append('Gauss')

        X = Xd.data
        validMask = 1.0 * (X < 1.E6)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        bx[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            bx.attrs[key.upper()]=vars()[key]

    bx.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    bx.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    bx.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    bx.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    bx.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (8. * divideFactor) * trgtAS / rsun_obs)
    bx.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    bx.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    bx.attrs['PIXLUNIT']  = list(pixlunit)

    # Process BY
    Xd = Map(filelist_by[-1])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            vars()[key] = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_by)):
        Xd = Map(file)
        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple' and key != 'history':
                vars()[key].append(Xd.meta[key])
        pixlunit.append('Gauss')

        X = Xd.data
        validMask = 1.0 * (X < 1.E6)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        by[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            by.attrs[key.upper()]=vars()[key]

    by.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    by.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    by.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    by.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    by.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (8. * divideFactor) * trgtAS / rsun_obs)
    by.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    by.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    by.attrs['PIXLUNIT']  = list(pixlunit)

    # Process BZ
    Xd = Map(filelist_bz[-1])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            vars()[key] = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_bz)):
        Xd = Map(file)
        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple' and key != 'history':
                vars()[key].append(Xd.meta[key])
        pixlunit.append('Gauss')

        X = Xd.data
        validMask = 1.0 * (X < 1.E6)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        bz[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple' and key != 'history':
            bz.attrs[key.upper()]=vars()[key]

    bz.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    bz.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    bz.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    bz.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    bz.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (8. * divideFactor) * trgtAS / rsun_obs)
    bz.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    bz.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    bz.attrs['PIXLUNIT']  = list(pixlunit)
