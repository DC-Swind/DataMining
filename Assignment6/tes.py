import numpy
from numpy import *
import random
import matplotlib.pyplot as plt

def init():
    indexs = random.sample(range(0,1000),8)
    print indexs
a = [[1,2,3],[2,4,6],[4,7,9]]
A = matrix(a)
print A
B = [[3]]
print B[0][0]

print [[0]*6]*2

x = matrix([1,2,3,4,5])
y = matrix([1,2,3,4,4])

d  = numpy.inner(x- y,x-y)
print d,d[0],d[0][0],int(d)
#print matrix(B)

init()


x = matrix([[1,2],[2,4],[4,7],[5,5]])
y = matrix([[1,2,3,4],[2,2,4,6],[5,4,7,9]])
print numpy.tensordot(x, y, axes=(0,-1))

print x + [1,2]
print numpy.sum(x,-1)
print x.shape[0],x.shape[1]
print multiply([1,2],[2,3])

print numpy.sum(x)

A = numpy.zeros((5,1))
B = numpy.zeros((5,5))

print B[:,0:2]

for rr in [50,25]:
    for i in range(1,40):
        sigma = i * 0.5
        print rr,sigma
        
