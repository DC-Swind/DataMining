from preprocessv2 import process
import numpy
import random
import time
from numpy import matrix
from PCA import PCA, PCA_Transform
import os
import csv
import matplotlib.pyplot as plt

def pegasos_log(x,y,lamda,T):
    dataN = len(x)
    featureN = len(x[0])
    w = numpy.zeros(featureN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)
        z = numpy.inner(w,x[i])
        expp = min(y[i] * z,100)
        w = ( 1 - yita * lamda ) * w + yita * (y[i] / (1 + numpy.exp(expp))) * x[i]    
    return w

def pegasos_hinge(x,y,lamda,T):
    dataN = len(x)
    featureN = len(x[0])
    w = numpy.zeros(featureN)
    
    """
    picx = []
    for i in range(20):
        picx.append(T/(20-i))
    picy = []
    index = 0
    """
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)

        if (y[i] * numpy.inner(w,x[i]) < 1):
            w = ( 1 - yita * lamda ) * w + yita * y[i] * x[i]
        else:
            w = ( 1 - yita * lamda ) * w
    """
        if index < 20 and picx[index] == t:
            right,error,callright,callerror,predict1,total1,max8i = test(x, y, w)
            picy.append(float(error)/(error + right))
            index += 1
    plt.plot(picx, picy)
    plt.xlabel("iterater times")
    plt.ylabel("loss")
    plt.show()
    """
    return w

def K(x1,x2):
    #return numpy.exp(-numpy.inner((x1-x2),(x1-x2))/(2*sigma2))
    return (numpy.inner(x1,x2)[0][0] + 1)**3

def pegasos_kernel(x,y,lamda,T):
    w = numpy.zeros(dataN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)

        yy = 0
        for j in range(dataN):
            yy += w[j] * y[j] * K(x[j],x[i])
        yy *= y[i] * yita
        if yy < 1:
            w[i] = w[i] + 1
    return w

def kerneltest(x,y,w):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    dataN = len(x)
    for i in range(dataN):
        z = 0
        for j in range(dataN):
            z += w[j]*y[j]*K(x[j],x[i])
        if (z * y[i] > 0):
            right = right + 1
            if y[i] == 1:
                callright += 1
        else:
            error = error + 1
            if y[i] == 1:
                callerror += 1
    return right,error,callright,callerror


def test(x,y,w):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    predict1 = 0
    total1 = 0
    dataN = len(x)
    max8i = 0.0
    for i in range(dataN):
        z = numpy.inner(x[i],w)
        
        if y[i] == 1:
            total1 += 1
        if z > 0:
            predict1 += 1
            if z > max8i:
                max8i = z
        if (z * y[i] > 0):
            right = right + 1
            if y[i] == 1:
                callright += 1
        else:
            error = error + 1
            if y[i] == 1:
                callerror += 1
    return right,error,callright,callerror,predict1,total1,max8i


def kernelize(x,l,sigma2):
    xN = len(x)
    lN = len(l)
    newx = numpy.zeros((xN,lN))
    #newx = []
    for i in range(lN):
        fi = x - l[i]
        newx[:,i] = numpy.exp(-numpy.sum(numpy.multiply(fi,fi),-1) /(2 * sigma2))
    return newx,lN


#main entry and setting
def main(x,y,dataN,featureN,sigma,ln):
    numpy.random.seed(seed=1) 
    lamda = 0.00003
    sigma2 = sigma

    
    cputime = time.time()
    #sample new features and kernelize
    lindex = random.sample(range(0,dataN),ln)

    l = numpy.zeros((ln,featureN))
    for i in range(len(lindex)):
        l[i,:] = x[lindex[i],:]

    x,featureN = kernelize(x,l,sigma2)
    print "kernelize",time.time() - cputime,"s"
    cputime = time.time()
    print dataN,featureN,sigma2



    #start svm
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
        #because +- is not equal, so choose 1/3 from -
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
        right,error,callright,callerror,predict1,total1,max8i = test(x,yy,w)
        if max8i == 0:
            max8i = 1
        max8[i] = max8i
    
    
        """
        w = pegasos_kernel(x,yy,lamda, 5 * dataN)
        right,error,callright,callerror = kerneltest(x,yy,w)
        """
        print "right:",right,"error:",error,"callright:",callright,"callerror:",callerror,"predict1",predict1,"total1",total1
    
        #print time.time() - cputime,"s"
        cputime = time.time()
    
    
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
            
    print "right rate:",str(right)+"/"+str(dataN),float(right)/dataN
    return w8,max8,l,float(right)/dataN




"""  Entry  """
#read file
pr = process()
x,y = pr.readtrainfile("train.csv")
dataN = x.shape[0]
featureN = x.shape[1]
print "data",dataN,"feature",featureN

#PCA
x, tfMatrix , recon = PCA(x, 100)
featureN = x.shape[1]

cputime = time.time()

#sigma 1.4 ~ 1.5
w8 = []
max8 = []
l = []
sigma = 0.0
maxrightrate = 0.0
for i in range(5):
    print "the",i+1,"rd time"
    r = random.random()
    w8i,max8i,li,rightrate = main(x,y,dataN,featureN,1.40 + r * 0.1,300)
    if rightrate > maxrightrate:
        maxrightrate = rightrate
        w8 = w8i
        max8 = max8i
        l = li
        sigma = 1.40 + r * 0.1




#test
ID,x = pr.readtestfile("test.csv")

#PCA Trans
x = PCA_Transform(x, tfMatrix)

cputime = time.time()
csvfile = file(os.path.join(os.getcwd(), "svmans.csv"),"wb")
writer = csv.writer(csvfile)
writer.writerow(["Id","Response"])

x,featureN = kernelize(x,l,sigma)

dataN = x.shape[0]
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