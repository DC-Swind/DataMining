import numpy as np
import random
import itertools
import time
import matplotlib
from blaze.expr.expressions import label
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from preprocessv1 import process
import os
import csv
from PCA import PCA, PCA_Transform
import xgboost as xgb
import pandas as pd

def buji(A, n):
    B = []
    index = 0
    for i in range(n):
        if index < len(A) and i == A[index]:
            index += 1
        else:
            B.append(i)
    return B

def getoutput(x, sp):
    
    if x < sp[0]:
        ans = 1
    elif x < sp[1]:
        ans = 2
    elif x < sp[2]:
        ans = 3
    elif x < sp[3]:
        ans = 4
    elif x < sp[4]:
        ans = 5
    elif x < sp[5]:
        ans = 6
    elif x < sp[6]:
        ans = 7
    else:
        ans = 8
        
    return ans

def printtable(pred, test_Y, sp):
    sos = []
    for i in range(8):
        sos.append([0] * 8)
    for i in range(len(pred)):
        sos[test_Y[i] - 1][getoutput(pred[i], sp) - 1] += 1
      
    print "axis-x is predict , axes-y is target"
    for i in range(8):
        for j in range(8):
            print sos[i][j],"\t",
        print ""
        
    prederror = sum( getoutput(pred[i], sp) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))
    print "errorrate",prederror
    
    return prederror   

def adjustsp(pred, test_Y, sp):
    newsp = [sp[i] for i in range(7)]
    
    for i in range(7):
        minerr = 9999999
        minspi = -1
        for j in range(41):
            if i > 0 and i + 0.5 + 0.05 * j < newsp[i-1]:
                continue 
            newsp[i] = i + 0.5 + 0.05 * j
            prederror = sum( getoutput(pred[k], newsp) != test_Y[k] for k in range(len(test_Y)) )
            if prederror < minerr:
                minerr = prederror
                minspi = newsp[i]
        newsp[i] = minspi
        print newsp[i],
    print ""
    prederror = sum( getoutput(pred[k], newsp) != test_Y[k] for k in range(len(test_Y)) )
    print prederror
    print sp
    prederror = sum( getoutput(pred[k], sp) != test_Y[k] for k in range(len(test_Y)) )
    print prederror
    return newsp
        
"""    Entry    """
#read file
pr = process()
trainx,trainy = pr.readtrainfile("train.csv")
"""
for i in range(trainx.shape[0]):
    trainy[i] -= 1
"""
dataN = trainx.shape[0]
featureN = trainx.shape[1]
print "data",dataN,"feature",featureN

testID,testx = pr.readtestfile("test.csv")

"""
ftrain = open("train.svm.txt","w+")
for i in range(trainx.shape[0]):
    ftrain.write(str(trainy[i]))
    for j in range(trainx.shape[1]):
        if trainx[i,j] != 0:
            ftrain.write(" "+str(j)+":"+str(trainx[i,j]))
    ftrain.write("\n")
ftrain.close()
ftest = open("test.svm.txt","w+")
for i in range(testx.shape[0]):
    ftest.write("8")
    for j in range(testx.shape[1]):
        if testx[i,j] != 0:
            ftest.write(" "+str(j)+":"+str(testx[i,j]))
    ftest.write("\n")
ftest.close()

train = xgb.DMatrix('train.svm.txt')
test = xgb.DMatrix('test.svm.txt') 
"""  

minerror = 1.0
bestxgb = -1
bestsetting = -1

deep = [8]
samp = [0.7]
#sp = [1.5,2.5,3.5,4.5,5.5,6.5,7.5]
sp = [1.9,3.5,3.5,4.5,5.5,6.3,6.8]
"""
for i in range(10):
    for j in range(10):
        deep.append(5 + i)
        samp.append(0.5 + j *0.05)
"""    
for cross in range(len(deep)):
    print "cross",cross+1,"setting",deep[cross],samp[cross]
    nb_test = trainx.shape[0]/80
    nb_train = trainx.shape[0] - nb_test
    samples_test = np.sort(random.sample(range(0,trainx.shape[0]),nb_test))
    samples_train = buji(samples_test, trainx.shape[0])
    
    
    train_X = trainx[samples_train]
    train_Y = [trainy[samples_train[i]] for i in range(len(samples_train))]
    test_X = trainx[samples_test]
    test_Y = [trainy[samples_test[i]] for i in range(len(samples_test))]
    
    xg_train = xgb.DMatrix(train_X, label = train_Y)
    xg_test = xgb.DMatrix(test_X, label = test_Y)
    
    #set parameter
    param = {}
    param['objective'] = 'reg:linear'
    #param['objective'] = 'multi:softmax'
    #param['class'] = 8
    # scale weight of positive examples
    param['eta'] = 0.05
    param['max_depth'] = deep[cross]
    param['silent'] = 1
    param['subsample '] = samp[cross]
    param['colsample_bytree'] = 0.3
    param['lambda'] = 10
    param['nthread'] = 8
    param['min_child_weight'] = 5
    param['early_stopping_rounds'] = 10
    num_round = 1400


    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    bst = xgb.train(param, xg_train, num_round, watchlist )

    pred = bst.predict( xg_test )
    
    prederror = printtable(pred, test_Y, sp)
    if prederror < minerror:
        minerror = prederror
        bestxgb = bst
        bestsetting = cross 
        
    #newsp = adjustsp(pred, test_Y, sp)
    #printtable(pred, test_Y, newsp)
    
    pred = bst.predict( xg_train )
    printtable(pred, train_Y, sp)
    
    #newsp = adjustsp(pred, train_Y, sp)
    #printtable(pred, train_Y, newsp)
    print "-----------------------------------------------"

print "best setting,deep:",deep[bestsetting],"samp:",samp[bestsetting],"errorrate:",minerror
"""    Output Ans    """
    
csvfile = file(os.path.join(os.getcwd(), "xgboostans.csv"),"wb")
writer = csv.writer(csvfile)
writer.writerow(["Id","Response"])
    
lenx = testx.shape[0]
xg_pred = xgb.DMatrix(testx)
pred = bestxgb.predict( xg_pred )
count = [0] * 8
for i in range(lenx):
    writer.writerow([testID[i],getoutput(pred[i] , sp)])
    count[getoutput(pred[i], sp) - 1] += 1
print count
csvfile.close()
