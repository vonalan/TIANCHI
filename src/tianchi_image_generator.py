import os 
import sys 

import numpy as np 
import sklearn.preprocessing as sklprep
from PIL import Image

import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp


def process_image(imagename, resultname, tmppath, params="--edge-thresh 6 --peak-thresh 3"):
    if imagename[-3:] != 'pgm':
        im = Image.open(imagename).convert('L')
        im.save(tmppath)
        # imagename ='tmp.pgm'
    cmmd = str("sift "+tmppath+" --output="+resultname+" "+params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)

def main(mode, path, pid=0, n_jobs=1): 
    # features = []

    for count, line in enumerate(tcdl.read_line_from_text(mode, path)): 
        if pid%n_jobs==0: 
            # feat = []

            _, idx, y, x = tcdl.process_line_from_text(mode, line)
            idx = float(idx)
            y = float(y)
            # feat += idx + y 
            
            x = [float(i) for i in x]
            x = np.array(x).reshape((15,4,101,101))
            x = np.transpose(x,(1,0,2,3))
            M,N,P,Q = x.shape

            for m in range(M): 
                for n in range(N): 
                    im = x[m,n,:,:]
                    img = Image.fromarray(im.astype(np.uint8))
                    imgdir = '../pic/%s_%d_%.2f'%(mode, idx, y)
                    if not os.path.exists(imgdir): 
                        os.makedirs(imgdir)
                    imgpath = '%s/%s_h%d_t%d.jpg'%(imgdir, imgdir[7:], m+1, n+1)
                    img.save(imgpath)

                    siftdir = '../sift/%s_%d_%.2f'%(mode, idx, y)
                    if not os.path.exists(siftdir): 
                        os.makedirs(siftdir)
                    siftpath = '%s/%s_h%d_t%d.txt'%(siftdir, siftdir[8:], m+1, n+1)
                    
                    tmpdir = '../tmp/%s_%d_%.2f'%(mode, idx, y)
                    if not os.path.exists(tmpdir): 
                        os.makedirs(tmpdir)
                    tmppath = '%s/%s_h%d_t%d.pgm'%(tmpdir, tmpdir[7:], m+1, n+1)

                    process_image(imgpath, siftpath, tmppath, params="--edge-thresh 6 --peak-thresh 3")
                    
            print(idx, y)
        break


mode = sys.argv[1]
pid = int(sys.argv[2])
n_jobs = int(sys.argv[3])

# main(mode, '../data/%s.txt'%mode, pid=pid, n_jobs=n_jobs)
main(mode, '../data/%s.txt'%mode, pid=pid, n_jobs=n_jobs)