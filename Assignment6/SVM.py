from preprocess import readfile
from preprocess import readtestfile
import numpy
import random
import time
from numpy import matrix
import os
import csv

def pegasos_log(x,y,lamda,T):
    w = numpy.zeros(featureN)
    dataN = len(x)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)
        z = numpy.inner(w,x[i])
        expp = min(y[i] * z,100)
        w = ( 1 - yita * lamda ) * w + yita * (y[i] / (1 + numpy.exp(expp))) * x[i]    
    return w

def pegasos_hinge(x,y,lamda,T):
    w = numpy.zeros(featureN)
    dataN = len(x)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)

        if (y[i] * numpy.inner(w,x[i]) < 1):
            w = ( 1 - yita * lamda ) * w + yita * y[i] * x[i]
        else:
            w = ( 1 - yita * lamda ) * w
        
    return w

def test(x,y,w,type):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    predict1 = 0
    total1 = 0
    dataN = len(x)
    for i in range(dataN):
        z = numpy.inner(x[i],w)
        
        if y[i] == 1:
            total1 += 1
        if z > 0:
            predict1 += 1
            if z > max8[type]:
                max8[type] = z
        if (z * y[i] > 0):
            right = right + 1
            if y[i] == 1:
                callright += 1
        else:
            error = error + 1
            if y[i] == 1:
                callerror += 1
    return right,error,callright,callerror,predict1,total1


#main entry and setting
numpy.random.seed(seed=1) 
lamda = 0.00005
sigma2 = 1

#read file
cputime = time.time()
x,y = readfile("train.csv")
print time.time() - cputime,"s"
cputime = time.time()
dataN = len(x)
featureN = len(x[0])
print dataN,featureN


w8 = []
max8 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#start svm
for i in range(8):
    print "start SVM",i+1,"    ",
    yy = []
    N1 = 0
    for t in y:
        if t == i+1:
            yy.append(1)
            N1 += 1
        else:
            yy.append(-1)
    
    xxx = numpy.zeros((dataN,featureN))
    index = 0
    yyy = []
    prob = float(N1)/(dataN - N1) * 4
    for j in range(dataN):
        if y[j] != i+1:
            s = random.random()
            if s < prob:
                yyy.append(-1)
                xxx[index,:] = x[j,:]
                index += 1
        else:
            yyy.append(1)
            xxx[index,:] = x[j,:]
            index += 1
            
    xxx = xxx[0:index,:]
    #w = pegasos_log(x,yy,lamda, 5 * dataN)
    w = pegasos_hinge(x,yy,lamda, 5 * dataN)
    w8.append(w)
    #print w
    right,error,callright,callerror,predict1,total1 = test(x,yy,w,i)
    
    
    """
    w = pegasos_kernel(x,yy,lamda, 5 * dataN)
    right,error,callright,callerror = kerneltest(x,yy,w)
    """
    print "right:",right,"error:",error,"callright:",callright,"callerror:",callerror,"predict1",predict1,"total1",total1
    
    #print time.time() - cputime,"s"
    cputime = time.time()
    

ID,x = readtestfile("test.csv")
print "read test file",time.time() - cputime,"s"
cputime = time.time()
csvfile = file(os.path.join(os.getcwd(), "ans.csv"),"wb")
writer = csv.writer(csvfile)
writer.writerow(["Id","Response"])
dataN = len(x)
for i in range(dataN):
    maxv = 0
    maxj = -1
    minv = -9999999.9
    minj = -1
    for j in range(8):
        if j == 6:
            continue
        z = numpy.inner(x[i],w8[j])#/max8[j]
        if (z > 0):
            if z > maxv:
                maxv = z
                maxj = j
        else:
            if z > minv:
                minv = z
                minj = j
                
    out = 8
    if maxj+1 > 0:
        out = maxj+1
    else:
        if minj+1 > 0:
            out = minj+1
    writer.writerow([ID[i],out])
csvfile.close()

"""
right = 0
for i in range(dataN):
    maxv = 0
    maxj = -1
    minv = -99999999.9
    minj = -1
    for j in range(8):
        if j == 6:
            continue
        z = numpy.inner(x[i],w8[j])#/max8[j]
        if (z > 0):
            if z > maxv:
                maxv = z
                maxj = j
        else:
            if z > minv:
                minv = z
                minj = j
    
    out = 8
    if maxj+1 > 0:
        out = maxj+1
    else:
        if minj+1 > 0:
            out = minj+1
    if out == y[i]:
        right += 1
            
print right
"""