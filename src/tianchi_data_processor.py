#-*- coding: UTF-8 -*- 


import random
import math 
import datetime 

# import loader
import utilities


def pre_define_processor(mode = ''):
    if mode == 'train' or mode == 'valid':
        path = utilities.getPath('train.txt')
        s1, s2 = 5, 6
        post = ''
    elif mode == 'testA':
        path = utilities.getPath('testA.txt')
        s1, s2 = 4, 6
        post = 'A'
    else:
        raise Exception('Value Empty Error! ')

    return path, s1, s2, post


def random_select_samples_2(m=10000, n1=10000, n2=2000):
    trainlist = [i for i in range(m)]
    validlist = [k for l in [range(j * 500, (j + 1) * 500) for j in [i for i in [0, 5, 10, 15]]] for k in l]
    testAlist = [i for i in range(n2)]
    print([len(trainlist), len(validlist), len(testAlist)])

    dict = {'train': trainlist, 'valid': validlist, 'testA': testAlist}
    return dict


def random_select_samples(m1 = 10000, m2 = 10000, m3 = 2000, n1 = 10000, n2 = 2000, n3 = 2000):
    samples_dict = {
        'train':random.sample([i + 1 for i in range(m1)], n1), 
        'valid':random.sample([i + 1 for i in range(m2)], n2), 
        'testA':random.sample([i + 1 for i in range(m3)], n3)
        }
    print([len(samples_dict['train']), len(samples_dict['valid']), len(samples_dict['testA'])])

    return samples_dict

def re_random_select_samples(sample_dict, num): 
    re_sample_dict = {
        'train':random.sample(sample_dict['train'], num), 
        'valid':random.sample(sample_dict['valid'], num), 
        'testA':random.sample(sample_dict['testA'], num)
        }
    print([len(re_sample_dict['train']), len(re_sample_dict['valid']), len(re_sample_dict['testA'])])

    return re_sample_dict




# Roulette 
def split(num = 10000, numtrain = 8000, numvalid = 2000):
    totalist = [i for i in range(num)]
    # random.shuffle(totalist) 
    cliptrain = random.sample(totalist, numtrain) 
    clipleft = [i for i in totalist if i not in cliptrain] # for the element of totls and clip is uniq
    clipvalid = random.sample(clipleft, numvalid) 
    clipleftleft = [i for i in clipleft if i not in clipvalid]
    
    # print cliptrain, clipvalid
    print(len(cliptrain), len(clipvalid), len(clipleftleft))
    return cliptrain, clipvalid

def sample2(prefix, cliptrain, clipvalid): 
    namestrain, namesvalid = [], [] 
    indextrain, indexvalid = [], [] 
    labelstrain, labelsvalid = [], [] 
    attrstrain, attrsvalid = [], []
    
    for i in range(20):
        path = utilities.getPath(prefix + '_' + str(i + 1) + '.cpk')
        obj = utilities.loadObject(path)
        names, index, labels, attrs = obj[0], obj[1], obj[2], obj[3]
        for idx in index: 
            j = idx - 1 - i * 500
            if j in cliptrain: 
                # print 'train: ', idx, j
                namestrain.append(names[j])
                indextrain.append(index[j])
                labelstrain.append(labels[j])
                attrstrain.append(attrs[j])
            elif j in clipvalid: 
                # print 'valid: ', idx, j
                namesvalid.append(names[j])
                indexvalid.append(index[j])
                labelsvalid.append(labels[j])
                attrsvalid.append(attrs[j])
            else: 
                pass 
    
    datatrain = [namestrain, indextrain, labelstrain, attrstrain]
    datavalid = [namesvalid, indexvalid, labelsvalid, attrsvalid]
    fntrain = utilities.getPath('train_2000_samples.cpk')
    fnvalid = utilities.getPath('valid_1000_samples.cpk')
    utilities.saveObjects([datatrain, datavalid], [fntrain, fnvalid])

	
def smaller(prefix, m, n): 
	for i in range(m): 
		fn1 = prefix + '_' + str(i + 1) + '.cpk'
		path = utilities.getPath(fn1)
		names, index, labels, attrs = utilities.loadObject(path)
		for j in range(n): 
			nms = names[j * 100:(j + 1) * 100]
			idx = index[j * 100:(j + 1) * 100]
			lbs = labels[j * 100:(j + 1) * 100]
			ats = attrs[j * 100:(j + 1) * 100]
			fn2 = prefix + '_' + 's' + '_ ' + str(i * 5 + (j + 1)) + '.cpk'
			path = utilities.getPath(fn2)
			object = [nms, idx, lbs, ats]
			utilities.saveObject(object, path)
			
			
def sample(prefix, nprefix, clip, num = 20): 
    namesset = [] 
    indexset = [] 
    lablesset = [] 
    attrsset = [] 
    
    for i in range(num):
        path1 = utilities.getPath(prefix + '_' + str(i + 1) + '.cpk')
        obj = utilities.loadObject(path1)
        names, index, labels, attrs = obj[0], obj[1], obj[2], obj[3]
        for idx in index: 
            j = idx - 1 - i * 500
            if j in clip: 
                # print nprefix, idx, j
                namesset.append(names[j])
                indexset.append(index[j])
                lablesset.append(labels[j])
                attrsset.append(attrs[j])
            else: 
                pass 
    
    dataset = [namesset,indexset,lablesset,attrsset]
    fn2 = nprefix + '_' + str(len(clip)) + '_' + 'samples.cpk'
    path2 = utilities.getPath(fn2)
    utilities.saveObject(dataset, path2)    
    
if __name__ == '__main__': 
    cliptrain, clipvalid = split(1000, 200, 100)
else: 
    pass 
