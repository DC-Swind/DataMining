import numpy as np
from numpy import argsort

def PCA(x,newfN):
    meanVals = np.mean(x, axis = 0)
    data  =  x - meanVals
    covMatrix = np.cov(data, rowvar = 0)
    eigVals, eigVects = np.linalg.eig(covMatrix)
    eigValInd = np.argsort(-eigVals, axis=0)  #sort , descend
    tfMatrix = eigVects[:,eigValInd[0:newfN]] #cut off unwanted dimension,and calculate transform matrix
    lowdimdata = np.dot(data,tfMatrix)
    reconData = np.dot(lowdimdata ,tfMatrix.T) + meanVals
    return lowdimdata, tfMatrix, reconData
    
def PCA_Transform(testx,tfMatrix):
    meanVals = np.mean(testx, axis = 0)
    data = testx - meanVals
    return np.dot(data,tfMatrix)
    
    