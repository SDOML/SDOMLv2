#!/usr/bin/python
# Convert the SDO/EVE .npy files of SDOML to .zarr files.

import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
import pandas as pd
import zarr
from numcodecs import Blosc, Delta


eve_irradiance = np.load('./EVE/irradiance.npy')
eve_jd = np.load('./EVE/jd.npy')
eve_logt = np.load('./EVE/logt.npy')
eve_name = np.load('./EVE/name.npy', allow_pickle=True)
eve_wavelength = np.load('./EVE/wavelength.npy')

t = Time(eve_jd, format='jd')
t.format = 'iso'

store = zarr.DirectoryStore('eve.zarr')
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
root = zarr.group(store=store,overwrite=True)
ints = root.create_group('MEGS-A')
time = ints.create_dataset('Time',shape=(np.shape(eve_irradiance)[0]),dtype='<U23',compressor=compressor)
fe18 = ints.create_dataset('Fe XVIII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe8 = ints.create_dataset('Fe VIII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe20 = ints.create_dataset('Fe XX',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe9 = ints.create_dataset('Fe IX',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe10 = ints.create_dataset('Fe X',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe11 = ints.create_dataset('Fe XI',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe12 = ints.create_dataset('Fe XII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe13 = ints.create_dataset('Fe XIII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe14 = ints.create_dataset('Fe XIV',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
he2 = ints.create_dataset('He II',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe15 = ints.create_dataset('Fe XV',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
he2_2 = ints.create_dataset('He II_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe16 = ints.create_dataset('Fe XVI',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe16_2 = ints.create_dataset('Fe XVI_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
mg9 = ints.create_dataset('Mg IX',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
s14 = ints.create_dataset('S XIV',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
ne7 = ints.create_dataset('Ne VII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
si12 = ints.create_dataset('Si XII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
si12_2 = ints.create_dataset('Si XII_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o3 = ints.create_dataset('O III',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
he1 = ints.create_dataset('He I',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o4 = ints.create_dataset('O IV',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe20_2 = ints.create_dataset('Fe XX_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
he1_2 = ints.create_dataset('He I_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe19 = ints.create_dataset('Fe XIX',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o3_2 = ints.create_dataset('O III_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
mg10 = ints.create_dataset('Mg X',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
mg10_2 = ints.create_dataset('Mg X_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o5 = ints.create_dataset('O V',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o2 = ints.create_dataset('O II',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
fe20_3 = ints.create_dataset('Fe XX_3',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
ne8 = ints.create_dataset('Ne VIII',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o4_2 = ints.create_dataset('O IV_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o2_2 = ints.create_dataset('O II_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
h1 = ints.create_dataset('H I',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
h1_2 = ints.create_dataset('H I_2',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
c3 = ints.create_dataset('C III',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
h1_3 = ints.create_dataset('H I_3',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)
o6 = ints.create_dataset('O VI',shape=(np.shape(eve_irradiance)[0]),dtype='f4',compressor=compressor)

time[:] = t.value

fe18[:] = eve_irradiance[:,0]
fe18.attrs['wavelength'] = str(eve_wavelength[0])+' nm'
fe18.attrs['logT'] = str(eve_logt[0])+' MK'
fe18.attrs['ion'] = 'Fe XVIII'

fe8[:] = eve_irradiance[:,1]
fe8.attrs['wavelength'] = str(eve_wavelength[1])+' nm'
fe8.attrs['logT'] = str(eve_logt[1])+' MK'
fe8.attrs['ion'] = 'Fe VIII'

fe20[:] = eve_irradiance[:,2]
fe20.attrs['wavelength'] = str(eve_wavelength[2])+' nm'
fe20.attrs['logT'] = str(eve_logt[2])+' MK'
fe20.attrs['ion'] = 'Fe XX'

fe9[:] = eve_irradiance[:,3]
fe9.attrs['wavelength'] = str(eve_wavelength[3])+' nm'
fe9.attrs['logT'] = str(eve_logt[3])+' MK'
fe9.attrs['ion'] = 'Fe IX'

fe10[:] = eve_irradiance[:,4]
fe10.attrs['wavelength'] = str(eve_wavelength[4])+' nm'
fe10.attrs['logT'] = str(eve_logt[4])+' MK'
fe10.attrs['ion'] = 'Fe X'

fe11[:] = eve_irradiance[:,5]
fe11.attrs['wavelength'] = str(eve_wavelength[5])+' nm'
fe11.attrs['logT'] = str(eve_logt[5])+' MK'
fe11.attrs['ion'] = 'Fe XI'

fe12[:] = eve_irradiance[:,6]
fe12.attrs['wavelength'] = str(eve_wavelength[6])+' nm'
fe12.attrs['logT'] = str(eve_logt[6])+' MK'
fe12.attrs['ion'] = 'Fe XII'

fe13[:] = eve_irradiance[:,7]
fe13.attrs['wavelength'] = str(eve_wavelength[7])+' nm'
fe13.attrs['logT'] = str(eve_logt[7])+' MK'
fe13.attrs['ion'] = 'Fe XIII'

fe14[:] = eve_irradiance[:,8]
fe14.attrs['wavelength'] = str(eve_wavelength[8])+' nm'
fe14.attrs['logT'] = str(eve_logt[8])+' MK'
fe14.attrs['ion'] = 'Fe XIV'

he2[:] = eve_irradiance[:,9]
he2.attrs['wavelength'] = str(eve_wavelength[9])+' nm'
he2.attrs['logT'] = str(eve_logt[9])+' MK'
he2.attrs['ion'] = 'He II'

fe15[:] = eve_irradiance[:,10]
fe15.attrs['wavelength'] = str(eve_wavelength[10])+' nm'
fe15.attrs['logT'] = str(eve_logt[10])+' MK'
fe15.attrs['ion'] = 'Fe XV'

he2_2[:] = eve_irradiance[:,11]
he2_2.attrs['wavelength'] = str(eve_wavelength[11])+' nm'
he2_2.attrs['logT'] = str(eve_logt[11])+' MK'
he2_2.attrs['ion'] = 'He II'

fe16[:] = eve_irradiance[:,12]
fe16.attrs['wavelength'] = str(eve_wavelength[12])+' nm'
fe16.attrs['logT'] = str(eve_logt[12])+' MK'
fe16.attrs['ion'] = 'Fe XVI'

fe16_2[:] = eve_irradiance[:,13]
fe16_2.attrs['wavelength'] = str(eve_wavelength[13])+' nm'
fe16_2.attrs['logT'] = str(eve_logt[13])+' MK'
fe16_2.attrs['ion'] = 'Fe XVI'

mg9[:] = eve_irradiance[:,14]
mg9.attrs['wavelength'] = str(eve_wavelength[14])+' nm'
mg9.attrs['logT'] = str(eve_logt[14])+' MK'
mg9.attrs['ion'] = 'Mg IX'

s14[:] = eve_irradiance[:,15]
s14.attrs['wavelength'] = str(eve_wavelength[15])+' nm'
s14.attrs['logT'] = str(eve_logt[15])+' MK'
s14.attrs['ion'] = 'S XIV'

ne7[:] = eve_irradiance[:,16]
ne7.attrs['wavelength'] = str(eve_wavelength[16])+' nm'
ne7.attrs['logT'] = str(eve_logt[16])+' MK'
ne7.attrs['ion'] = 'Ne VII'

si12[:] = eve_irradiance[:,17]
si12.attrs['wavelength'] = str(eve_wavelength[17])+' nm'
si12.attrs['logT'] = str(eve_logt[17])+' MK'
si12.attrs['ion'] = 'Si XII'

si12_2[:] = eve_irradiance[:,18]
si12_2.attrs['wavelength'] = str(eve_wavelength[18])+' nm'
si12_2.attrs['logT'] = str(eve_logt[18])+' MK'
si12_2.attrs['ion'] = 'Si XII'

o3[:] = eve_irradiance[:,19]
o3.attrs['wavelength'] = str(eve_wavelength[19])+' nm'
o3.attrs['logT'] = str(eve_logt[19])+' MK'
o3.attrs['ion'] = 'O III'

he1[:] = eve_irradiance[:,20]
he1.attrs['wavelength'] = str(eve_wavelength[20])+' nm'
he1.attrs['logT'] = str(eve_logt[20])+' MK'
he1.attrs['ion'] = 'He I'

o4[:] = eve_irradiance[:,21]
o4.attrs['wavelength'] = str(eve_wavelength[21])+' nm'
o4.attrs['logT'] = str(eve_logt[21])+' MK'
o4.attrs['ion'] = 'O IV'

fe20_2[:] = eve_irradiance[:,22]
fe20_2.attrs['wavelength'] = str(eve_wavelength[22])+' nm'
fe20_2.attrs['logT'] = str(eve_logt[22])+' MK'
fe20_2.attrs['ion'] = 'Fe XX'

he1_2[:] = eve_irradiance[:,23]
he1_2.attrs['wavelength'] = str(eve_wavelength[23])+' nm'
he1_2.attrs['logT'] = str(eve_logt[23])+' MK'
he1_2.attrs['ion'] = 'He I'

fe19[:] = eve_irradiance[:,24]
fe19.attrs['wavelength'] = str(eve_wavelength[24])+' nm'
fe19.attrs['logT'] = str(eve_logt[24])+' MK'
fe19.attrs['ion'] = 'Fe XIX'

o3_2[:] = eve_irradiance[:,25]
o3_2.attrs['wavelength'] = str(eve_wavelength[25])+' nm'
o3_2.attrs['logT'] = str(eve_logt[25])+' MK'
o3_2.attrs['ion'] = 'O III'

mg10[:] = eve_irradiance[:,26]
mg10.attrs['wavelength'] = str(eve_wavelength[26])+' nm'
mg10.attrs['logT'] = str(eve_logt[26])+' MK'
mg10.attrs['ion'] = 'Mg X'

mg10_2[:] = eve_irradiance[:,27]
mg10_2.attrs['wavelength'] = str(eve_wavelength[27])+' nm'
mg10_2.attrs['logT'] = str(eve_logt[27])+' MK'
mg10_2.attrs['ion'] = 'Mg X'

o5[:] = eve_irradiance[:,28]
o5.attrs['wavelength'] = str(eve_wavelength[28])+' nm'
o5.attrs['logT'] = str(eve_logt[28])+' MK'
o5.attrs['ion'] = 'O V'

o2[:] = eve_irradiance[:,29]
o2.attrs['wavelength'] = str(eve_wavelength[29])+' nm'
o2.attrs['logT'] = str(eve_logt[29])+' MK'
o2.attrs['ion'] = 'O II'

fe20_3[:] = eve_irradiance[:,30]
fe20_3.attrs['wavelength'] = str(eve_wavelength[30])+' nm'
fe20_3.attrs['logT'] = str(eve_logt[30])+' MK'
fe20_3.attrs['ion'] = 'Fe XX'

ne8[:] = eve_irradiance[:,31]
ne8.attrs['wavelength'] = str(eve_wavelength[31])+' nm'
ne8.attrs['logT'] = str(eve_logt[31])+' MK'
ne8.attrs['ion'] = 'Ne VIII'

o4_2[:] = eve_irradiance[:,32]
o4_2.attrs['wavelength'] = str(eve_wavelength[32])+' nm'
o4_2.attrs['logT'] = str(eve_logt[32])+' MK'
o4_2.attrs['ion'] = 'O IV'

o2_2[:] = eve_irradiance[:,33]
o2_2.attrs['wavelength'] = str(eve_wavelength[33])+' nm'
o2_2.attrs['logT'] = str(eve_logt[33])+' MK'
o2_2.attrs['ion'] = 'O II'

h1[:] = eve_irradiance[:,34]
h1.attrs['wavelength'] = str(eve_wavelength[34])+' nm'
h1.attrs['logT'] = str(eve_logt[34])+' MK'
h1.attrs['ion'] = 'H I'

h1_2[:] = eve_irradiance[:,35]
h1_2.attrs['wavelength'] = str(eve_wavelength[35])+' nm'
h1_2.attrs['logT'] = str(eve_logt[35])+' MK'
h1_2.attrs['ion'] = 'H I'

c3[:] = eve_irradiance[:,36]
c3.attrs['wavelength'] = str(eve_wavelength[36])+' nm'
c3.attrs['logT'] = str(eve_logt[36])+' MK'
c3.attrs['ion'] = 'C III'

h1_3[:] = eve_irradiance[:,37]
h1_3.attrs['wavelength'] = str(eve_wavelength[37])+' nm'
h1_3.attrs['logT'] = str(eve_logt[37])+' MK'
h1_3.attrs['ion'] = 'H I'

o6[:] = eve_irradiance[:,38]
o6.attrs['wavelength'] = str(eve_wavelength[38])+' nm'
o6.attrs['logT'] = str(eve_logt[38])+' MK'
o6.attrs['ion'] = 'O VI'
