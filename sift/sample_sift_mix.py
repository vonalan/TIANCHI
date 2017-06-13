# /DenseTrackStab ./testA_1_-1.00_h4.avi >> halo.txt -L 10 -W 4 -N 16 -s 2 -t 2


import os 
import sys 
import subprocess
import random

import numpy as np

global sifts
global count_total

sifts = [] 
count_total = 0

def sampling(mode, fact=10): 
    '''bug'''
    video_list = '../data/common/video-list-%s.txt'%mode

    # pid = int(sys.argv[2])
    # num_tasks = int(sys.argv[3])
    pid = 0
    num_tasks = 1
    
    rf1 = open(video_list, 'r')
    videos = rf1.readlines()
    rf1.close()

    videos = [video.rstrip() for video in videos]
    # videos = ['testA_1000_-1.00_h4.avi', 'testA_1002_-1.00_h4.avi']
    # videos = random.sample(videos,len(videos))

    # [info, traj, hof, hog, mbhx, mbhy] = [10, 20, 64, 72, 64, 64]

    index, labels, cline = [], [], []

    for i in range(len(videos)): 
        if i%num_tasks == pid:
            for j in range(15): 
                # dir_bin = './DenseTrackStab'
                # dir_vid = '../vid/%s'%videos[i]
                dir_idt = '../sift/%s/%s_t%d.txt'%(videos[i][:-7], videos[i][:-4], j+1)
                # print(dir_idt)
                # options = '-L 10 -W 4 -N 16 -s 2 -t 2'
                
                idx = float((videos[i]).strip().split('_')[1])
                label = float((videos[i]).strip().split('_')[2])
                
                global sifts
                global count_total
                
                if os.path.exists(dir_idt):
                    rf2 = open(dir_idt,'r')
                    count = 0
                    for line in rf2.readlines():
                        count_total += 1
                        if count_total%fact == 0: 
                            sline = line.strip().split()
                            fline = [float(e) for e in sline[4:]]
                            sifts += [fline]
                            count += 1
                        # print fline
                        # break
                    index += [idx]
                    labels += [label]
                    cline += [count]

                    rf2.close()
                    print(len(cline), len(labels), count, len(sifts), len(sifts[0]))

                else:
                    print('%s is not existed! '%dir_idt)

        # break
        
    indexA = np.array(index).reshape((-1,1))
    labelsA = np.array(labels).reshape((-1,1))
    clineA = np.array(cline).astype(int).reshape(-1,1)

    np.savetxt('../data/index_%s_%s_f%d.txt'%(mode, 'mix', fact), indexA)
    np.savetxt('../data/lables_%s_%s_f%d.txt'%(mode, 'mix', fact), labelsA)
    np.savetxt('../data/cline_%s_%s_f%d.txt'%(mode, 'mix', fact), clineA)
    print(indexA.shape, labelsA.shape, clineA.shape)

# if __name__ == '__main__': 
fact = 10

sampling('train', fact=fact)
sampling('testA', fact=fact)

# global sifts
siftsA = np.array(sifts)
print(siftsA.shape)
np.savetxt('../data/sifts_%s_%s_f%d.txt'%('ramdom', 'mix', fact), siftsA)