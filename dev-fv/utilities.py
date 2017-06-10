import platform 
import datetime 
import pickle
import numpy


def getPath_2(filename):
    path = ""
    
    getPlatform = platform.system()
    if 'Windows' in getPlatform:
        path = "..\data\\" + filename 
    elif 'Linux' in getPlatform:
        path = "../data/" + filename 
    else:
        pass
    
    return path


def getPath(filename):
    path = ""

    getPlatform = platform.system()
    if 'Windows' in getPlatform:
        path = "..\data\\" + filename
    elif 'Linux' in getPlatform:
        path = "../data/" + filename
    else:
        pass

    return path


def getDir():
    dir = ""

    getPlatform = platform.system()
    if 'Windows' in getPlatform:
        dir = "D:/Users/kingdom/TIANCHI"
    elif 'Linux' in getPlatform:
        dir = "/home/kingdom/TIANCHI"
    else:
        pass

    return dir


def fetchTime():
    return str(datetime.datetime.now())

def saveMatrix(matrixName, matrixFile):
    numpy.save(matrixFile, matrixName)
    print(str(datetime.datetime.now())[:19] + " " + matrixFile+ " is saved! shape: " + str(numpy.shape(matrixName)))


def saveMatrices(matrixNames, matrixFiles): 
    for matrixName, matrixFile in zip(matrixNames, matrixFiles):
        numpy.save(matrixFile, matrixName)
        print(str(datetime.datetime.now())[:19] + " " + matrixFile+ " is saved! shape: " + str(numpy.shape(matrixName)))

        
def loadMatrix(matrixFile): 
    matrix = numpy.load(matrixFile)
    print(str(datetime.datetime.now())[:19] + " " + matrixFile+ " is loaded! shape: " + str(numpy.shape(matrix)))
    return matrix 


def loadMatrices(matrixFiles): 
    matrices = [] 
    for matrixFile in matrixFiles: 
        matrix = numpy.load(matrixFile)
        matrices.append(matrix)
        print(str(datetime.datetime.now())[:19] + " " + matrixFile+ " is loaded! shape: " + str(numpy.shape(matrix)))
    return matrices


def saveObject(objectName, objectFile):     
    with open(objectFile, "wb") as File:
        pickle.dump(objectName, File)
    
    print(str(datetime.datetime.now())[:19] + " " + objectFile + " is saved! ")


def saveObjects(objectNames, objectFiles): 
    for objectName, objectFile in zip(objectNames, objectFiles): 
        with open(objectFile, "wb") as File:
            pickle.dump(objectName, File)
        
        print(str(datetime.datetime.now())[:19] + " " + objectFile + " is saved! ")


def loadObject(filename):
    object = ()
    with open(filename, 'rb') as rf:
        # names, index, labels, attrs = pickle.load(rf)
        object = (pickle.load(rf))
        print(str(datetime.datetime.now())[:19] + " " + filename + " is loaded! ")
    return object

    
def loadObjects(filenames):
    objects = []
    for fn in filenames:
        with open(fn, 'rb') as rf:
            names, index, labels, attrs = pickle.load(rf)
            obj = [names, index, labels, attrs]
            objects.append(obj)
            print(str(datetime.datetime.now())[:19] + " " + fn + " is loaded! ")
    return objects


print(fetchTime() + " implemeting the application in the platform: " + platform.system())