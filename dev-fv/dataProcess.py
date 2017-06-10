#-*- coding: UTF-8 -*- 

import datetime
import numpy
import dataProcess 
import scipy.spatial.distance as sciDist
import sklearn.cluster as sklCluster
import sklearn.neighbors as sklNeighbors

def genDeltaX(Q, trainX, number = 1000):
    return numpy.mat(numpy.random.uniform(-Q, Q, (number, numpy.shape(trainX)[1])))

def pDist(dataSet):
    return sciDist.pdist(dataSet)

def cDist(dataSet1, dataSet2):
    return sciDist.cdist(dataSet1, dataSet2)

def fv2iv(dataSet):
    pass

def ftoi(dataSet):
    ftoiM = numpy.mat(numpy.zeros(numpy.shape(dataSet)))
    maxC = numpy.max(dataSet, axis = 1)
    ftoiM[dataSet == maxC] = 1
    if (numpy.sum(ftoiM, axis = 1) == 1).all() == False: 
        distM = numpy.mat(dataProcess.cDist(dataSet, dataSet)) 
        maxT = numpy.max(distM) 
        for i in xrange(numpy.shape(dataSet)[0]): 
            distM[i,i] = 2 * maxT 
        idxR = numpy.argmin(distM, axis = 1) 
        idxC = numpy.nonzero(numpy.sum(ftoiM, axis = 1) != 1)[0]
        ftoiM[idxC] = ftoiM[idxR[idxC]]
    assert (numpy.sum(ftoiM, axis = 1) == 1).all() == True
    return ftoiM

def vtoi(dataSet):
    pass

def kMeans(dataSet, k):
    kmResult = sklCluster.KMeans(n_clusters=k, random_state=0).fit(dataSet)
    centroids = kmResult.cluster_centers_

    return centroids 

def centroidAssign(centroids, dataSet):
    errM = numpy.power(numpy.mat(cDist(dataSet, centroids)), 2)
    minIdxR, minR = numpy.argmin(errM, axis = 1), numpy.min(errM, axis = 1)
    caNew = numpy.concatenate((minIdxR, minR), axis = 1)

    return caNew


def BOW(clusterAssessment, countLine, bins):
    featureHists = []
    begin = 0
    end = 0
    for i in xrange(len(countLine)):
        end = begin + countLine[i]
        featureHist = numpy.histogram(clusterAssessment[begin:end, 0], bins, range = (0, bins))[0]
        featureHists.append(featureHist)
        begin = end
    return numpy.mat(featureHists)


def numCenterAssign(numCenter, trainY):
    ct = numpy.mat(numpy.sum(trainY, axis = 0))
    cn = numpy.mat(numpy.ones(numpy.shape(ct)),dtype=int)
    numCenter = numCenter - numpy.shape(cn)[1]

    for i in range(numCenter):
        res = ct/cn
        idx = numpy.mat(numpy.nonzero(res[0,:] == numpy.max(res[0,:], axis = 1))[1])[0,0]
        cn[0,idx] = cn[0,idx] + 1 # is cn proportional to number of every catogory in TrainY ?!!

    return cn


def innerCluster_2(trainX, trainY, numCenter, alpha, numClass=8):
    kms = sklCluster.KMeans(n_clusters=numClass, random_state=0)
    kms.fit(trainY)
    # centroids = kms.cluster_centers_
    labels = (kms.labels_).reshape(trainY.shape)

    # int2vec()
    vec_lab = numpy.zeros((labels.shape[0], numClass))
    for i in range(labels.shape[0]):
        vec_lab[i, labels[i, 0]] = 1

    assert (vec_lab.sum(axis=1) == 1).all()


    cn = numCenterAssign(numCenter, vec_lab)
    U = numpy.mat(numpy.zeros((numCenter, numpy.shape(trainX)[1])))
    '''define V'''
    V = numpy.mat(numpy.zeros((numCenter, 1)))
    '''define V'''
    centerLabel = numpy.mat(numpy.zeros((numCenter, 1)))

    begin, end = 0, 0
    for i in range(numpy.shape(cn)[1]):
        end = begin + cn[0, i]
        centerLabel[0, begin:end] = i
        centroidsBest = kMeans(trainX[numpy.nonzero(vec_lab[:, i])[0]], int(cn[0, i]))
        U[begin:end, :] = centroidsBest
        '''bug bug bug bug'''
        '''the definition of radius of RBFNN'''
        '''bug bug bug bug'''
        '''compute V'''
        distM = cDist(trainX[numpy.nonzero(vec_lab[:, i])[0]], U[begin:end, :])
        # distM = cDist(trainX, U[begin:end, :])
        distMaxR = numpy.transpose(numpy.mat(numpy.max(distM, axis = 0)))
        V[begin:end,:] = distMaxR * alpha
        # distMeanR = numpy.transpose(numpy.mat(numpy.mean(distM, axis = 0)))
        # V[begin:end,:] = distMeanR * alpha
        # distM = pDist(trainX[numpy.nonzero(trainY[:, i])[0]])
        # distMeanT = numpy.mean(distM)
        # V[begin:end, :] = numpy.tile(distMeanT * alpha, (cn[0, i], 1))
        '''compute V'''
        begin = end

    assert begin == numCenter
    return U, V


def innerCluster(trainX, trainY, numCenter, alpha, numClass=5):
    # cn = numCenterAssign(numCenter, trainY)
    # U = numpy.mat(numpy.zeros((numCenter, numpy.shape(trainX)[1])))
    U = kMeans(trainX, numCenter)

    '''define V'''
    distM = cDist(trainX, U)
    distMaxR = numpy.transpose(numpy.mat(numpy.max(distM, axis=0)))
    V = distMaxR * alpha
    '''define V'''

    return U, V


def nonzero_count(dataSet, axis = -1):
    if axis == 0:
        countList = numpy.mat(numpy.zeros((numpy.shape(dataSet)[axis], 1)))
        for i in range(numpy.shape(dataSet)[axis]):
            countList[i,0] = numpy.count_nonzero(dataSet[i,:])
        return countList
    elif axis == 1: 
        countList = numpy.mat(numpy.zeros((1, numpy.shape(dataSet)[axis])))
        for i in range(numpy.shape(dataSet)[axis]):
            countList[0,i] = numpy.count_nonzero(dataSet[:,i])
        return countList
    elif axis == -1: 
        return numpy.count_nonzero(dataSet) 
    else: 
        # print "value error! reset to default: axis = -1. "
        return numpy.count_nonzero(dataSet) 

def autoNorm(dataSet, countLine, axis = ""):
    fCountLine = countLine.astype(float)
    tf = dataSet/fCountLine
    if axis == "TF":
        return tf 
    elif axis == "TFIDF":
        idf = numpy.log(numpy.shape(dataSet)[0]/(nonzero_count(dataSet, axis = 1) + 1))
        return numpy.multiply(tf, idf)
    else:
        maxFQ = numpy.max(dataSet, axis = 1)
        fMaxFQ = maxFQ.astype(float)
        return dataSet/fMaxFQ




def sigmoid(weightedSum): 
    return 1.0/(1.0 + numpy.exp(weightedSum))

def sigmoid_prime(weightedSum):
    sp = numpy.multiply(sigmoid(weightedSum), (1 - sigmoid(weightedSum)))
    return sp
    # return numpy.multiply(sigmoid(weightedSum), (1 - sigmoid(weightedSum))) # warch out! Hadamard product!!!

# !!! redefine feedforward function for RBFNN 
def feedforwardRBFNN(trainX, trainY, W, U, V):
    D = numpy.power(dataProcess.cDist(trainX, U), 2)
    Z = numpy.exp(D/numpy.transpose((-2 * numpy.power(V, 2)))) # how to implement element-wise operation?
    A = numpy.dot(Z, W)
    assert numpy.shape(A) == numpy.shape(trainY)
    return A 

# !!! redefine feedforward function for MLPNN 
def feedforwardMLPNN(weights, biases, activation):
    for weight, bias in zip(weights, biases):
        activation = dataProcess.sigmoid(numpy.dot(weight, activation) + bias) # bias, bug!!!
    return activation


if __name__ == '__main__':
    pass
else:
    # print str(datetime.datetime.now())[:19] + " importing module dataProcess ... "
    pass


# end 
