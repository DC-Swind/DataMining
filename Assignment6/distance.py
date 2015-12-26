from preprocessv1 import readfile
from preprocessv1 import readtestfile
import time
import numpy
from numpy import matrix
import os
import csv

def distance(x,y):
    x = matrix(x)
    y = matrix(y)
    return float(numpy.inner(x-y,x-y))


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

#calculate means
m = []
for i in range(9):
    m.append([0] * featureN)
mn = [0] * 9
for i in range(dataN):
    for j in range(featureN):
        m[y[i]][j] = (m[y[i]][j] * mn[y[i]] + x[i][j]) / (mn[y[i]] + 1)
    mn[y[i]] += 1
    
print time.time() - cputime,"s"
cputime = time.time()


#read test file
ID,x = readtestfile("test.csv")
print time.time() - cputime,"s"
cputime = time.time()

csvfile = file(os.path.join(os.getcwd(), "ans.csv"),"wb")
writer = csv.writer(csvfile)
writer.writerow(["Id","Response"])
ans = []
dataN = len(x)
for i in range(dataN):
    mind = 9999999.9
    minj = -1
    for j in range(8):
        dis = distance(x[i],m[j+1])
        if dis < mind:
            mind = dis
            minj = j+1
    ans.append([ID[i],minj])
    writer.writerow([ID[i],minj])
#print ans
csvfile.close()

