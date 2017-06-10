import os 
import sys 

import numpy as np

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
    fv_type = ['D:/Users/kingdom/TIANCHI/fv/%s.%s.fv.txt'%(videos_fv, t) for t in ['traj', 'hog', 'hof', 'mbhx', 'mbhy']]
    v_path = 'D:/Users/kingdom/TIANCHI/tmp/%s'%videos_fv
    
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
    pathslist = ['D:/Users/kingdom/TIANCHI/fv/%s.%s.fv.txt'%(video, fvtype) for fvtype in ['traj','hof','hog','mbhx','mbhy']]
    for path in pathslist:
        for line in read_line_from_text('train', path):
            sline = line.lstrip().rstrip().split()
            fline = [float(s) for s in sline]
            print(len(fline))
            fvlist += fline
    print(len(fvlist))
    
    # fvm = (np.array(fvlist).reshape((-1,1))).T
    # np.savetxt('%s.fv.txt'%(video), fvm)

    return fvlist

def iter_mini_batch(mode, sample_dict, mini_batch_size= 100):
    videoList = 'D:/Users/kingdom/TIANCHI/data/video-list-%s.txt'%mode
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
                print('mergeing for %s ... '%vname)
                mfvm = merge_fv(videos[i][7:])

                mini_mode += [mmode]
                mini_idx += [midx]
                mini_label += [mlab]
                mini_fvm += [mfvm]
            else:
                # pass
                print('fv for %s has not existed! '%(videos[i]))

            count += 1

        if count == mini_batch_size or count == left:
            mini_mode_mat = np.array(mini_mode).reshape((-1,1))
            mini_idx_mat = np.array(mini_idx).reshape((-1,1))
            mini_lab_mat = np.array(mini_label).reshape((-1,1))
            mini_fvm_mat = np.array(mini_fvm)
            print(mini_mode_mat.shape, mini_idx_mat.shape, mini_lab_mat.shape, mini_fvm_mat.shape)
            if(mini_idx_mat.shape[0]):
                yield mini_mode_mat, mini_idx_mat, mini_lab_mat, mini_fvm_mat
            else:
                pass

            mini_mode = []
            mini_idx = []
            mini_fvm = []
            mini_label = []

            left = left - count
            count = 0

for _, _, labels, features in iter_mini_batch('train', [i for i in range(7000,8000)]):
    print(labels, features)