# /DenseTrackStab ./testA_1_-1.00_h4.avi >> halo.txt -L 10 -W 4 -N 16 -s 2 -t 2

import os 
import sys 
import subprocess
import random

import numpy as np
import sklearn.cluster as sklcluster
from sklearn.externals import joblib


def build_hist(mode, inmode, f, k): 
    video_list = '../data/common/video-list-%s.txt'%mode

    num_tasks = 1
    pid = 0

    rf1 = open(video_list, 'r')
    videos = rf1.readlines()
    rf1.close()


    videos = [video.rstrip() for video in videos] # [:3]
    # videos = ['train_7880_0.00_h4.avi', 'train_7478_20.00_h4.avi']
    # [info, traj, hof, hog, mbhx, mbhy] = [10, 20, 64, 72, 64, 64]

    HIST = np.zeros((0,k),dtype=int)

    for i in range(len(videos)): 
        if i%num_tasks == pid:
            H = np.zeros((15,k))
            for j in range(15): 
                # dir_bin = './DenseTrackStab'
                # dir_vid = '../vid/%s'%videos[i]
                dir_idt = '../sift/%s/%s_t%d.txt'%(videos[i][:-7], videos[i][:-4], j+1)
                # options = '-L 10 -W 4 -N 16 -s 2 -t 2'

                sifts = []
                if os.path.exists(dir_idt):
                    rf2 = open(dir_idt,'r')
                    for line in rf2.readlines():
                        sline = line.strip().split()
                        fline = [float(e) for e in sline[4:]]
                        sifts += [fline]

                    rf2.close()
                else:
                    print('%s is not existed! '%dir_idt)
                
                h = np.zeros((1,k), dtype=int)
                if len(sifts): 
                    siftsA = np.array(sifts)
                    # print siftsA 
                    x = kms.predict(siftsA)
                    # print x
                    h = (np.histogram(x, bins=k, range=(0,k))[0]).astype(int)
                H[j,:] = h
            HIST = np.vstack((HIST, H.reshape((1,15*k))))
                
        break

    np.savetxt('../data/hist_%s_%s_f%d_k%d.txt'%(mode, inmode, f, k), HIST)


type = 'sifts'
mode = 'random'
inmode = 'mix'

for f in [10]:
    # mode = 'train'
    # fact = 25
    # '''bug bug bug'''

    # X = np.loadtxt('../data/%s_%s_%s_f%d.txt'%(type, mode, inmode, f)) # [:10000,:]
    # print('../data/%s_%s_%s_f%d.txt'%(type, mode, inmode, f), X.shape)

    for k in [256]:
        # for k in [4096,8192]:
        # '''bug bug bug'''
        # k = 2048
        # kms = sklcluster.KMeans(n_clusters=k, n_init=8, random_state=None, n_jobs=-1-8-8-4-2)
        # kms.fit(X)
        # print('../data/%s_kmeans_%s_f%d_k%d.model'%(type, inmode, f, k))
        # joblib.dump(kms, '../data/%s_kmeans_%s_f%d_k%d.model'%(type, inmode, f, k))
        kms = joblib.load('../data/%s_kmeans_%s_f%d_k%d.model'%(type, inmode, f, k))
        # build_hist(mode, inmode, f, k)
        build_hist('train', 'random', f, k)
        build_hist('testA', 'random', f, k)