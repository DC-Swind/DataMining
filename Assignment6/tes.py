import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
import itertools
def init():
    indexs = random.sample(range(0,1000),8)
    print indexs


print [[0]*6]*2

x = matrix([1,2,3,4,5])
y = matrix([1,2,3,4,4])

d  = np.inner(x- y,x-y)
print d,d[0],d[0][0],int(d)
#print matrix(B)

init()

x = matrix([[1,2],[2,4],[4,7],[5,5]])
y = matrix([[1,2,3,4],[2,2,4,6],[5,4,7,9]])
print np.tensordot(x, y, axes=(0,-1))

print x + [1,2]
print np.sum(x,-1)
print x.shape[0],x.shape[1]
print multiply([1,2],[2,3])

print np.sum(x)

B = matrix([[1,2,3],[2,3,4],[3,4,5]])

print B
print B[[0,2]]

A = matrix((1,2))

list = []
list.append(A)
list.append(B)

print "list",list[0] 
list[0] += 1
print list[0]
print A
A = np.zeros((1,2))
A[0,0] = 2
A[0,1] = 3
print A
print A**2