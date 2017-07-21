import numpy as np 
import sklearn.preprocessing as sklprep

import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp


# x = (np.random.randint(0,255,(1,15*4*101*101))).reshape(15,4,101,101)
# y = np.random.normal(loc=25, scale=20, size=(1,1))


def cuboid(x=None, center=(7,1,50,50), radius=(1,1,1,1)): 
    shape = x.shape

    ub = [c+r+1 for c, r in zip(center, radius)]
    lb = [c-r for c, r in zip(center, radius)]

    ub = [i if i <= u else u for i,u in zip(ub, shape)]
    lb = [0 if i< 0 else i for i in lb]

    # the cuboid centred at center with radius of radius, and
    # the statistics of the cuboid_4d: sum, min, max, range, mean, std
    cub = x[lb[0]:ub[0],lb[1]:ub[1],lb[2]:ub[2],lb[3]:ub[3]]
    sta = [cub.sum(), cub.min(), cub.max(), cub.max() - cub.min(), cub.mean(), cub.std()] 

    # print(sta)
    return sta


def main(mode, path, fact=1): 
    features = []

    for count, line in enumerate(tcdl.read_line_from_text(mode, path)): 
        if not count%fact: 
            feat = []

            _, idx, y, x = tcdl.process_line_from_text(mode, line)
            idx = [float(idx)]
            y = [float(y)]
            feat += idx + y 
            
            x = [float(i) for i in x]
            x = np.array(x).reshape((15,4,101,101))
            M,N,P,Q = x.shape

            feat += cuboid(x, center=(7,1,50,50), radius=(1,1,2,2))
            for i in range(M): 
                feat += cuboid(x, center=(i,1,50,50), radius=(1,1,2,2))
            
            print(feat[:8])
            features += [feat]
        
        # break
    
    # return np.array(features)
    np.savetxt('stas_%s.txt'%mode, features)


main('train', '../data/train.txt', fact=1)
main('testA', '../data/testA.txt', fact=1)