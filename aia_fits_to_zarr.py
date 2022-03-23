#Given: 
#   a folder --target to contain zarr files
#   an integer --scale (a proper divisor of 1024) containing the target output size
#Converts the source fits files to target zarr files:
#   -Rescaling the sun to a constant pixel size and correcting for invalid interpolation
#   -Applying the degradation constant and accounting for exposure time
#   -Then downsampling by mean
#   -Save all fits header information to meta data in zarr

import os, pdb
import numpy as np
import sunpy.io
from tqdm import tqdm
import zarr
from sunpy.map import Map
import skimage.transform
import gcsfs
from numcodecs import Blosc, Delta
import warnings
import argparse

warnings.filterwarnings("ignore")

CHANNELS = [131,1600,1700,171,193,211,304,335,94,4500]
trgtAS = 976.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',dest='target',required=True)
    parser.add_argument('--scale',dest='scale',required=True,type=int)
    args = parser.parse_args()
    return args

def loadAIADegrads(path):
    #return wavelength -> (date -> degradation dictionary)                                         
    degrads = {}
    for wl in CHANNELS:
        degrads[wl] = getDegrad("%s/degrad_%d.csv" % (path,wl))
    return degrads

def getDegrad(fn):
    #map YYYY-MM-DD -> degradation parameter                                                      
    lines = open(fn).read().strip().split("\n")
    degrad = {}
    for l in lines:
        d, f = l.split(",")
        degrad[d[1:11]] = float(f)
    return degrad

if __name__ == "__main__":
    args = parse_args()

    degrads = loadAIADegrads("degrad/")
    print(np.shape(degrads))

    print(args)
    target, scale = args.target, args.scale

    if not os.path.exists(target):
        os.mkdir(target)

    divideFactor = np.int(1024 / scale)

    #load file lists for AIA channels
    filelist_131  = np.load('./filelists_2011/filelist_131.npy')
    filelist_1600 = np.load('./filelists_2011/filelist_1600.npy')
    filelist_1700  = np.load('./filelists_2011/filelist_1700.npy')
    filelist_171  = np.load('./filelists_2011/filelist_171.npy')
    filelist_193  = np.load('./filelists_2011/filelist_193.npy')
    filelist_211  = np.load('./filelists_2011/filelist_211.npy')
    filelist_304  = np.load('./filelists_2011/filelist_304.npy')
    filelist_335  = np.load('./filelists_2011/filelist_335.npy')
    filelist_94  = np.load('./filelists_2011/filelist_94.npy')

    store = zarr.DirectoryStore(target+'sdomlv2_2011.zarr')
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=store,overwrite=True)
    year = root.create_group('2011')
    aia131 = year.create_dataset('131A',shape=(np.shape(filelist_131)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia1600 = year.create_dataset('1600A',shape=(np.shape(filelist_1600)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia1700 = year.create_dataset('1700A',shape=(np.shape(filelist_1700)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia171 = year.create_dataset('171A',shape=(np.shape(filelist_171)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia193 = year.create_dataset('193A',shape=(np.shape(filelist_193)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia211 = year.create_dataset('211A',shape=(np.shape(filelist_211)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia304 = year.create_dataset('304A',shape=(np.shape(filelist_304)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia335 = year.create_dataset('335A',shape=(np.shape(filelist_335)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    aia94 = year.create_dataset('94A',shape=(np.shape(filelist_94)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)

    # Process 131A
    Xd = Map(filelist_131[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_131)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia131[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia131.attrs[key.upper()]=vars()[key]

    aia131.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia131.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia131.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia131.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia131.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia131.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia131.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia131.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia131.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia131.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia131.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 1600A
    Xd = Map(filelist_1600[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_1600)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia1600[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia1600.attrs[key.upper()]=vars()[key]
    
    aia1600.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia1600.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia1600.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia1600.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia1600.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia1600.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia1600.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia1600.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia1600.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia1600.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia1600.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 1700A
    Xd = Map(filelist_1700[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_1700)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia1700[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia1700.attrs[key.upper()]=vars()[key]

    aia1700.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia1700.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia1700.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia1700.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia1700.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia1700.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia1700.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia1700.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia1700.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia1700.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia1700.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 171A
    Xd = Map(filelist_171[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_171)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia171[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia171.attrs[key.upper()]=vars()[key]

    aia171.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia171.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia171.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia171.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia171.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia171.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia171.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia171.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia171.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia171.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia171.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 193A
    Xd = Map(filelist_193[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_193)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia193[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia193.attrs[key.upper()]=vars()[key]
    
    aia193.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia193.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia193.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia193.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia193.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia193.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia193.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia193.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia193.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia193.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia193.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 211A
    Xd = Map(filelist_211[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_211)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia211[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia211.attrs[key.upper()]=vars()[key]
    
    aia211.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia211.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia211.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia211.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia211.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia211.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia211.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia211.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia211.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia211.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia211.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 304A
    Xd = Map(filelist_304[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_304)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia304[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia304.attrs[key.upper()]=vars()[key]
    
    aia304.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia304.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia304.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia304.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia304.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia304.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia304.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia304.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia304.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia304.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia304.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 335A
    Xd = Map(filelist_335[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_335)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia335[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia335.attrs[key.upper()]=vars()[key]
    
    aia335.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia335.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia335.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia335.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia335.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia335.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia335.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia335.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia335.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia335.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia335.attrs['PIXLUNIT']  = list(pixlunit)

    # Process 94A
    Xd = Map(filelist_94[0])
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            vars()[key] = []
    deg_cor = []
    pixlunit = []

    for fn,file in tqdm(enumerate(filelist_94)):
        Xd = Map(file)
        fn2 = file[-26:].split("_")[0].replace("AIA","")
        datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
        wavelength = int(file[-26:].split("_")[-1].replace(".fits",""))
        correction = degrads[wavelength][datestring]

        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                vars()[key].append(Xd.meta[key])
        deg_cor.append(correction)
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        expTime = max(Xd.meta['EXPTIME'],1e-2)
        rad = Xd.meta['RSUN_OBS']
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = Xr / (expTime*correction)
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
        Xr = Xr.astype('float32')
        aia94[fn,:,:]=Xr

    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            aia94.attrs[key.upper()]=vars()[key]

    aia94.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    aia94.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
    aia94.attrs['CDELT1'] = list(np.asarray(cdelt1, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia94.attrs['CDELT2'] = list(np.asarray(cdelt2, dtype=np.float64) * divideFactor * rsun_obs / trgtAS)
    aia94.attrs['R_SUN'] = list(np.asarray(r_sun, dtype=np.float64) / (4. * divideFactor) * trgtAS / rsun_obs)
    aia94.attrs['X0_MP'] = list(np.asarray(x0_mp, dtype=np.float64) / (4. * divideFactor))
    aia94.attrs['Y0_MP'] = list(np.asarray(y0_mp, dtype=np.float64) / (4. * divideFactor))
    aia94.attrs['CRPIX1']  = list(np.asarray(crpix1, dtype=np.float64) / divideFactor)
    aia94.attrs['CRPIX2']  = list(np.asarray(crpix2, dtype=np.float64) / divideFactor)
    aia94.attrs['DEG_COR']  = list(np.asarray(deg_cor, dtype=np.float64))
    aia94.attrs['PIXLUNIT']  = list(pixlunit)
