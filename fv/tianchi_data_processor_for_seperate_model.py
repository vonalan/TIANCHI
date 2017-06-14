import numpy as np
import pickle
import sklearn.linear_model as scilm
import sklearn.metrics as scimtc
import sklearn.metrics.pairwise as scidist

import utilities
# import loader
import tianchi_data_loader as tcdl 
import tianchi_data_processor as tcdp 


def binary_label(labels, threshold):
    bin_labels = labels.copy()
    for i in range(bin_labels.shape[0]):
        if bin_labels[i,0] < threshold:
            bin_labels[i,0] = -1
        else:
            bin_labels[i, 0] = 1
    return bin_labels


def tranform_feature(features, num, shape = (-1, 4 * 101 * 101)):
    return np.reshape(features, shape)


def duplicate_label(labels, num):
    dup_labels = np.tile(labels, (1, num))
    dup_labels = np.reshape(dup_labels, (-1,1))

    return dup_labels


def fixed_centroids(mode):
    pass


def calc_centroid(mode, split_list_train):
    idx = 0
    count = 0
    sumarr = np.zeros((1, 15 * 4 * 101 * 101), dtype=np.float)
    for _, attrsA, labelsA in tcdl.iter_mini_batches(mode, split_list_train, mini_batch_size = 500):
        print('haha')
        count += attrsA.shape[0]
        sumarr += attrsA.sum(axis = 0)
        idx += 1
        print('calc_centroid: %d, count: %d'%(idx, count))
        print(sumarr)
        ''''''
        # break
    return sumarr/float(count)


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
        split_dict[int(i)] = sort_list[i][:,0].tolist()
    
    return split_dict


def split_valid_data_batch_mode(mode, samplelist, centroid_lt_20, centroid_ge_20):
    lt_20_list, ge_20_list, similarlist = [], [], []
    index, labels, simcos, disecl, dismht = [], np.zeros((0,1)), np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,2))

    centroids = np.concatenate((centroid_lt_20, centroid_ge_20), axis=0)
    for batch, (idx, attrsA, labelsA) in enumerate(tcdl.iter_mini_batches(mode, samplelist, mini_batch_size=500)):
        index += idx
        labels = np.concatenate((labels, labelsA), axis = 0)
        # simcos = np.concatenate((simcos, scidist.cosine_similarity(attrsA, centroids)), axis = 0)
        # disecl = np.concatenate((disecl, scidist.euclidean_distances(attrsA, centroids)), axis=0)
        dismht = np.concatenate((dismht, scidist.manhattan_distances(attrsA, centroids)), axis=0)
        print("split_valid_data_batch_mode: I'm alive! batch: %d"%(batch + 1))
        ''''''
        # break

    # simsum = simcos/np.tile(simcos.sum(axis=1).reshape((-1,1)), (1, 2))
    # dissum = 1 - disecl/np.tile(disecl.sum(axis=1).reshape((-1,1)), (1, 2))
    dismhtsum = 1 - dismht / np.tile(dismht.sum(axis=1).reshape((-1, 1)), (1, 2))
    # weight = (simsum + dissum)/2
    weight = dismhtsum

    for i in range(len(index)):
        if weight[i,0] < weight[i,1]:
            ge_20_list.append(labels[i, 0])
        else:
            lt_20_list.append(labels[i, 0])
        print([mode, index[i], labels[i, 0], weight[i,0], weight[i,1]])
        similarlist.append([mode, index[i], labels[i, 0], weight[i,0], weight[i,1]])

    print('split_valid_data_batch_mode for %s: %d, %d' % (mode, len(lt_20_list), len(ge_20_list)))
    return lt_20_list, ge_20_list, similarlist


def split_valid_data(mode, samplelist, centroid_lt_20, centroid_ge_20):
    path, _, _, _ = tcdp.pre_define_processor(mode=mode)

    # kmeans clustering
    # cosine similarity
    simlist = []
    ge_20_list = [] 
    lt_20_list = []

    for lineFile in tcdl.read_line_from_text(mode, path):
        name, idx, label, attr = tcdl.process_line_from_text(mode, lineFile)
        if idx - 1 in samplelist:
            dist_1 = scidist.cosine_similarity(attr, centroid_lt_20)
            dist_2 = scidist.cosine_similarity(attr, centroid_ge_20)
            weight = dist_1/(dist_1 + dist_2)
            if dist_1 < dist_2: 
                ge_20_list.append(idx)
                print('%s, ge_20: %d, label: %f, weight: %f'%(mode, idx, label, 1 - weight))
            else: 
                lt_20_list.append(idx)
                print('%s, lt_20: %d, label: %f, weight: %f'%(mode, idx, label, weight))
            simlist.append([idx, label, weight, 1 - weight])
        ''''''
        # break
    return lt_20_list, ge_20_list, simlist


def multi_cosine_similarity(attrsX, mode, split_list):
    count = 0 
    sum_dist = 0 
    for attrsA, _ in tcdl.iter_mini_batches(mode, split_list, mini_batch_size = 100):
        count += attrsA.shape[0]
        print("shape: %d", attrsA.shape[0])
        sum_dist += (scidist.cosine_similarity(attrsX, attrsA)).sum()
        ''''''
        # break
    return sum_dist/float(count)


def split_train_data(mode, samplelist): 
    path, _, _, _ = tcdp.pre_define_processor(mode = mode)

    lt_20_list = []
    ge_20_list = []
    for lineFile in tcdl.read_line_from_text(mode, path): 
        name, idx, label, attr = tcdl.process_line_from_text(mode, lineFile)
        if idx - 1 in samplelist:
            if label >= 20:
                ge_20_list.append(idx)
                print('%s, ge_20: %d, label: %f, weight: %f' % (mode, idx, label, 1))
            else:
                lt_20_list.append(idx)
                print('%s, lt_20: %d, label: %f, weight: %f' % (mode, idx, label, 1))
    
    return lt_20_list, ge_20_list


def split_valid_data_2(mode, samplelist, train_lt_20_list, train_ge_20_list):
    # print('hahahah')
    path, _, _, _ = tcdp.pre_define_processor(mode=mode)
    # kmeans clustering
    # cosine similarity
    simlist = []
    ge_20_list = [] 
    lt_20_list = []
    for lineFile in tcdl.read_line_from_text(mode, path):
        name, idx, label, attr = tcdl.process_line_from_text(mode, lineFile)
        if idx - 1 in samplelist:
            print('valid: %d'%(idx))
            dist_1 = multi_cosine_similarity(attr, mode, train_lt_20_list)
            dist_2 = multi_cosine_similarity(attr, mode, train_ge_20_list)
            weight = dist_1/(dist_1 + dist_2)
            if dist_1 < dist_2: 
                ge_20_list.append(idx)
                print('ge_20: %d, label: %f, weight: %f'%(idx, label, 1 - weight))
            else: 
                lt_20_list.append(idx)
                print('lt_20: %d, label: %f, weight: %f'%(idx, label, weight))
            simlist.append([idx, label, weight, 1 - weight])
        ''''''
        # break
    return lt_20_list, ge_20_list, simlist


def split_selected_samples(iter, sample_dict):
    split_dict_global = {
        'train': {}, 
        'valid': {}, 
        'testA': {}
    }

    return split_dict_global


def load_splited_selected_samples():
     idx_ys_centroids_train = np.loadtxt('../data/idx_ys_centroids_train.csv', delimiter=',')
     split_dict = split_train_data_batch_mode('train', idx_ys_centroids_train)
     return split_dict


if __name__ =='__main__': 
    # idx_ys_centroids_train = np.loadtxt('../data/idx_ys_centroids_train.csv', delimiter=',')
    split_dict = load_splited_selected_samples()
    for i in range(3):
        print(split_dict[i])