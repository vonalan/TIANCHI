import pickle
import random 

import numpy as np
import sklearn.linear_model as scilm
import sklearn.metrics as scimtc
import sklearn.metrics.pairwise as scidist

import utilities
# import loader
import tianchi_data_loader as tcdl 
import tianchi_data_processor as tcdp 
import tianchi_data_cluster as tcdc

def shuffle(mlist): 
    return random.sample(mlist, len(mlist)) 
    

def split_train_data_batch_mode(mode, samplelist):
    centroids = set(samplelist[:, 2])
    # print(centroids)

    split_list = list()
    for i in centroids: 
        arr = samplelist[samplelist[:, 2] == i] 
        split_list.append(arr)
        # print(arr)
    sort_list = sorted(split_list, key = lambda x : np.mean(x[:,2]))

    split_dict = dict()
    for i in range(len(centroids)):
        split_dict[int(i)] = shuffle(sort_list[i][:,0].tolist())
    
    return split_dict


def split_valid_data_batch_mode(mode, samplelist): 
    pass 


def split_data_global(): 
    pass 


def load_splitted_data(): 
    pass 








def argument(list2, num): 
    diff = num - len(list2)
    if diff > 0: 
        list2 += random.sample(list2, diff)
    else: 
        list2 = random.sample(list2, num)

    return shuffle(list2)


def merge(list1, list2, num =40):
    list3 = [] 

    list1, list2 = shuffle(list1), shuffle(list2)
    list1, list2 = argument(list1, num), argument(list2, num)

    j, k = 0, 0
    for i in range(2 * num): 
        if i%2: 
            list3.append(list1[j]) 
            j += 1 
        else: 
            list3.append(list2[k])  
            k += 1

    return list3



def merge_2(lt_list, ge_list): 
    m_list = [] 
    j, k = 0, 0
    for i in range(len(lt_list) + len(ge_list)): 
        if i%5 in [0, 2, 4]: 
            m_list.append(lt_list[j]) 
            j += 1 
        elif i%5 in [1, 3]: 
            m_list.append(ge_list[k])  
            k += 1
        else: 
            pass 

    return m_list

if __name__ =='__main__': 
    lt_list, ge_list = tcdp.split(num = 100, numtrain = 60, numvalid = 40)
    print(lt_list[:40])
    print(ge_list[:40])

    # lt_list, ge_list = shuffle(lt_list), shuffle(ge_list)
    print(lt_list[:40])
    print(ge_list[:40])

    m_list = merge(lt_list, ge_list)
    print(m_list)