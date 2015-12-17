from preprocess import readfile
import numpy
import time
from svmutil import *

#main entry and setting
numpy.random.seed(seed=1) 
lamda = 0.00005
sigma2 = 1
"""
#read file
cputime = time.time()
x,y = readfile("train.csv")
print time.time() - cputime,"s"
cputime = time.time()
dataN = len(x)
featureN = len(x[0])
print dataN,featureN

#write file
fw = open("libsvm_data",'w')
for i in range(dataN):
    if y[i] == 1:
        fw.write("+1")
    else:
        fw.write("-1")
        
    for j in range(featureN):
        if x[i][j] != -1.0:
            fw.write(" "+str(j+1)+":"+str(x[i][j]))
    fw.write("\n")
fw.close()
"""
y, x = svm_read_problem('libsvm_data')
m = svm_train(y[:200], x[:200])
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

print p_acc
