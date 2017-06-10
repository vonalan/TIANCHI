import os 
import sys 
import random 

import numpy as np

import utilities as utils

def read_line_from_text(mode, path):
    # path = utilities.getPath(path)
    rf = open(path, 'r')

    while True:
        line = rf.readline()
        if line:
            yield line
        else:
            break

    rf.close()

def check_fv(videos_fv):
    '''bug bug bug'''
    # typelist = ['traj', 'hof', 'hog', 'mbhx', 'mbhy']
    typelist = ['traj']
    # typelist = ['hof', 'hog']
    # typelist = ['mbhx', 'mbhy']

    dir = utils.getDir()
    fv_type = ['%s/fv/%s.%s.fv.txt'%(dir, videos_fv, t) for t in typelist]
    # v_path = 'D:/Users/kingdom/TIANCHI/tmp/%s'%videos_fv
    
    flag_1 = 1
    for t in fv_type: 
        if not os.path.exists(t): 
            flag_1 = 0
            break
    
    flag_2 = 1
    # if os.path.exists(v_path):
    #     flag_2 = 1
    
    flag = flag_1 and flag_2
    
    
    # if not flag:
    #     # print 'false', v_path
    #     for t in fv_type:
    #         if os.path.exists(t):
    #             print('remove %s'%(t))
    #             os.remove(t)
    #     if os.path.exists(v_path):
    #         print(v_path)
    #         os.remove(v_path)
    #
    #     return False
    #
    # return True

    return flag

def merge_fv(video): 
    fvlist = []
    '''bug bug bug'''
    # typelist = ['traj', 'hof', 'hog', 'mbhx', 'mbhy']
    typelist = ['traj']
    # typelist = ['hof', 'hog']
    # typelist = ['mbhx', 'mbhy']

    dir = utils.getDir()
    pathslist = ['%s/fv/%s.%s.fv.txt' % (dir, video, fvtype) for fvtype in typelist]
    for path in pathslist:
        for line in read_line_from_text('train', path):
            sline = line.lstrip().rstrip().split()
            fline = [float(s) for s in sline]
            # print(len(fline))
            fvlist += fline
    # print(len(fvlist))
    
    # fvm = (np.array(fvlist).reshape((-1,1))).T
    # np.savetxt('%s.fv.txt'%(video), fvm)

    return fvlist

def iter_mini_batches(mode, sample_dict, mini_batch_size= 100):
    videoList = '%s/data/video-list-%s.txt'%(utils.getDir(), mode)
    # outputBase = 'D:/Users/kingdom/TIANCHI/fv/'
    totalTasks = 1
    pID = 0

    mini_mode = []
    mini_idx = []
    mini_fvm = []
    mini_label = []

    f = open(videoList, 'r')
    videos = f.readlines()
    f.close()
    videos = [video.rstrip() for video in videos]

    # shuffle is important to sgd algorithm 
    videos = random.sample(videos, len(videos))

    count = 0
    left = len(sample_dict) - count
    for i in range(len(videos)):
        if left == 0:
            break

        # if i % totalTasks == int(pID):
        vname = videos[i][7:]
        sname = vname.rstrip().lstrip().split('_')
        mmode, midx, mlab = sname[0], int(sname[1]), float(sname[2][:-4])

        if midx in sample_dict:
            if check_fv(vname):
                # print('mergeing fisher vectors for %s ... '%vname)
                mfvm = merge_fv(videos[i][7:])

                mini_mode += [mmode]
                mini_idx += [midx]
                mini_label += [mlab]
                mini_fvm += [mfvm]
            else:
                pass
                # print('fv for %s has not existed! '%(videos[i]))

            count += 1

        if count == mini_batch_size or count == left:
            mini_mode_mat = np.array(mini_mode).reshape((-1,1))
            mini_idx_mat = np.array(mini_idx).reshape((-1,1))
            mini_lab_mat = np.array(mini_label).reshape((-1,1))
            mini_fvm_mat = np.array(mini_fvm)
            # print(mini_mode_mat.shape, mini_idx_mat.shape, mini_lab_mat.shape, mini_fvm_mat.shape)
            print([mini_idx_mat.shape[0], mini_fvm_mat.min(), mini_fvm_mat.mean(), mini_fvm_mat.max(),
                  mini_lab_mat.min(), mini_lab_mat.mean(), mini_lab_mat.max()])
            if(mini_idx_mat.shape[0]):
                yield mini_idx_mat, mini_fvm_mat, mini_lab_mat
            else:
                pass

            mini_mode = []
            mini_idx = []
            mini_fvm = []
            mini_label = []

            left = left - count
            count = 0

if __name__ == '__main__':
    for _, _, labels, features in iter_mini_batches('train', [i for i in range(7000,8000)]):
        print(labels, features)