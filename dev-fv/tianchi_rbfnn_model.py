from sklearn.metrics import mean_squared_error as sklmse

import datetime
import numpy
import dataProcess
# import evaluate
import tianchi_data_processor as tcdp
import tianchi_data_loader as tcdl



class RBFNN:
    # initialize the archietecture for Neural Network
    def __init__(self, indim=30720, numCenter=128, outdim=1, alpha=1.0, Q=0.05, numClass=4):
        self.len_traj = 30720
        self.len_hof = 110592
        self.len_hog = 98304
        self.len_mbhx = 98304
        self.len_mbhy = 98304

        self.indim = self.len_traj
        self.numCenter = numCenter
        self.outdim = outdim
        self.alpha = alpha
        self.Q = Q
        self.numClass = numClass
        # self.ration = ration 
        # self.nRound = nRound 
        # self.indim = indim # K
        # self.numCenter = numCenter # M 
        # self.outdim = outdim # category 
        self.U = numpy.mat([numpy.random.uniform(-1, 1, indim) for j in range(numCenter)])
        self.V = numpy.mat([[0.25] for j in range(numCenter)])
        self.W = numpy.mat(numpy.random.random([self.numCenter, self.outdim]))
        '''new'''
        # self.alpha = alpha 
        # self.Q = Q
        # self.trainx = trainx ##############################################
        # self.testx = testx ################################################
        # self.trainy = trainy ##############################################
        # self.testy = testy ################################################
        # self.trainout = trainout 
        # self.testout = testout 
        # self.confusedmatrix = confusedmatrix ##############################
        '''new'''


    # calculate activations
    def activCalc(self, trainX):
        Z = numpy.power(dataProcess.cDist(trainX, self.U), 2)
        A = numpy.exp(Z/numpy.transpose((-2 * numpy.power(self.V, 2)))) # how to implement element-wise operation?
        assert numpy.shape(A) == (numpy.shape(trainX)[0], self.numCenter)
        return A


    # calculate the parameters of hidden neurons in Network
    def centersCalc(self, trainX, trainY):
        self.U, self.V = dataProcess.innerCluster(trainX, trainY, self.numCenter, self.alpha, numClass=self.numClass)
        '''bug bug bug bug'''
        '''alpha'''
        '''bug bug bug bug'''
        assert numpy.shape(self.U) == (self.numCenter, numpy.shape(trainX)[1])
        # print str(datetime.datetime.now())[:19] + " The U calculation is done! The shape of the U is: ", numpy.shape(self.U)
        # self.V = numpy.tile(numpy.mean(dataProcess.pDist(trainX)) * alpha, (self.numCenter, 1))
        assert numpy.shape(self.V) == (self.numCenter, 1)
        # print str(datetime.datetime.now())[:19] + " The V calculation is done! The shape of the V is: ", numpy.shape(self.V)


    # classifier training
    def train(self, trainX, trainY):
        self.centersCalc(trainX, trainY)
        H = self.activCalc(trainX)
        self.W = numpy.dot(numpy.linalg.pinv(H), trainY)
        assert numpy.shape(self.W) == (self.numCenter, numpy.shape(trainY)[1])

        return self.valid(trainX, trainY)


    # classifier testing
    def valid(self, trainX, trainY):
        H = self.activCalc(trainX)
        output = numpy.dot(H, self.W)
        assert numpy.shape(output) == (numpy.shape(trainX)[0], self.outdim)
        rmse = numpy.sqrt(sklmse(output, trainY))

        return output, rmse


rbfnn = RBFNN(numCenter=512, numClass=8)
sample_dict = tcdp.random_select_samples()

mode = 'train'
for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=8000)):
    y, rmse = rbfnn.train(features, labels)
    # np.savetxt('ys_train_lsq.txt', np.hstack((index, labels, y)))
    print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
    break

mode = 'valid'
for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches('train', sample_dict[mode], mini_batch_size=2000)):
    y, rmse = rbfnn.valid(features, labels)
    numpy.savetxt('ys_valid_lsq.txt', numpy.hstack((index, labels, y)))
    print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
    break

mode = 'testA'
for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=2000)):
    y, rmse = rbfnn.valid(features, labels)
    numpy.savetxt('ys_testA_lsq.txt', numpy.hstack((index, labels, y)))
    print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
    break

print('\n')