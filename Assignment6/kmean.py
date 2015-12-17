from preprocess import readfile
import random
import numpy
from numpy import matrix
import time


def init_means(x,k):
    dN = len(x)
    indexs = random.sample(range(0,dN),k)
    m = []
    for i in indexs:
        m.append(x[i])
    return m

def distance(x,y):
    x = matrix(x)
    y = matrix(y)
    return float(numpy.inner(x-y,x-y))

def equal(x,y):
    l = len(x)
    for i in range(l):
        if x[i] != y[i]:
            return False
    return True

def assign(x,m):
    k = len(m)
    dN = len(x)
    newm = [[0] * len(m[0])]*k
    newmn = [0] * k
    for i in range(dN):
        mind = 9999999.9
        minj = -1
        for j in range(k):
            dis = distance(x[i],m[j])
            if (dis < mind):
                mind = dis
                minj = j
        
        for j in range(len(m[0])):
            newm[minj][j] = (newm[minj][j]*newmn[minj] + x[i][j])/(newmn[minj] + 1)
        newmn[minj] += 1
    return newm

def kmeans(x,k):
    m = init_means(x,k)
    cputime = time.time()
    itern = 1
    while True:
        newm = assign(x,m)
        
        if equal(m,newm) == True:
            break
        
        m = newm
        
        print itern,time.time() - cputime,"s"
        cputime = time.time()
        itern += 1
       
    return m
#main entry and setting
numpy.random.seed(seed=1) 
lamda = 0.00005
sigma2 = 1

#read file
cputime = time.time()
x,y = readfile("train.csv")
print time.time() - cputime,"s"


m = kmeans(x,8)

