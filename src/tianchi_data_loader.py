#-*- coding: UTF-8 -*-


import numpy 
import pickle
import datetime
import random


import utilities
import tianchi_data_processor as tcdp


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


def process_line_from_text(mode, lineFile):
    _, s1, s2, post = tcdp.pre_define_processor(mode = mode)

    strLine = lineFile.strip().split(',')
    info, label, attr = strLine[0], strLine[1], strLine[2]

    vname = info[:s1] + post
    idx = int(info[s2:])
    fltlabel = float(label)
    curattr = attr.strip().split()

    return vname, idx, fltlabel, curattr


def iter_mini_batches(mode, dict, mini_batch_size = 100):
    path, s1, s2, post = tcdp.pre_define_processor(mode)
    names, index, labels, attrs = [], [], [], []

    count = 0
    left = len(dict) - count 
    for lineFile in read_line_from_text(mode, path):
        if left == 0:
            break

        strLine = lineFile.strip().split(',')
        info, label, attr = strLine[0], strLine[1], strLine[2]

        vname = info[:s1] + post
        idx = int(info[s2:])
        fltlabel = float(label)
        curattr = attr.strip().split()

        ''''''
        if idx in dict:
            names.append(vname)
            index.append(idx)
            labels.append(fltlabel)
            attrs.append(curattr)
            count += 1
            # print('count: %d, left: %d'%(count, left))

        if count == mini_batch_size or count == left:
            indexA = numpy.array(index).reshape((-1, 1))
            attrsA = numpy.array(attrs).astype(numpy.float32).reshape((-1, 15 * 4 * 101 * 101))
            labelsA = numpy.array(labels).astype(numpy.float32).reshape((-1, 1))
            amin, amean, amax, lmin, lmean, lmax = attrsA.min(), attrsA.mean(), attrsA.max(), labelsA.min(), labelsA.mean(), labelsA.max()
            print(numpy.shape(labelsA)[0], [amin, amean, amax, lmin, lmean, lmax])
            # attrsA = attrsA / 255.0
            yield indexA, attrsA, labelsA

            names, index, labels, attrs = [], [], [], []
            left = left - count
            count = 0

    print('traversing the file %s is complete! '%(path))