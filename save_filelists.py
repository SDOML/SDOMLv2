#!/usr/bin/python
#Save file lists for the AIA channels.

import os, pdb
import numpy as np
import sunpy.io
import warnings
import argparse

warnings.filterwarnings("ignore")

CHANNELS = [131,1600,1700,171,193,211,304,335,94,4500]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print(args)
    src = args.src

    if not os.path.exists(target):
        os.mkdir(target)

    filelist_131=[]
    filelist_1600=[]
    filelist_1700=[]
    filelist_171=[]
    filelist_193=[]
    filelist_211=[]
    filelist_304=[]
    filelist_335=[]
    filelist_94=[]
    filelist_4500=[]

    for root, dirs, files in os.walk(src,followlinks=True):
        for file in tqdm(files):
            if file.endswith(".fits") and int(file[-12:-10]) % 6 == 0:
                if file[-9:-5] == '0131':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_131.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '1600':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_1600.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '1700':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_1700.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0171':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_171.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0193':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_193.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0211':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_211.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0304':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_304.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0335':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_335.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                elif file[-9:-5] == '0094':
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_94.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue
                else:
                    try:
                        Xh = sunpy.io.read_file_header(os.path.join(root,file))
                        if Xh[1]['QUALITY'] == 0:
                            filelist_4500.append(os.path.join(root,file))
                    except:
                        print("FILE CORRUPTED: %s" % file)
                        continue

np.save('filelist_131.npy',filelist_131)
np.save('filelist_1600.npy',filelist_1600)
np.save('filelist_1700.npy',filelist_1700)
np.save('filelist_171.npy',filelist_171)
np.save('filelist_193.npy',filelist_193)
np.save('filelist_211.npy',filelist_211)
np.save('filelist_304.npy',filelist_304)
np.save('filelist_335.npy',filelist_335)
np.save('filelist_94.npy',filelist_94)
np.save('filelist_4500.npy',filelist_4500)
