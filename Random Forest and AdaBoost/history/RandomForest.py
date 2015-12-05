import numpy
import math
import random
from numpy import sort
import DecisionTree as DT

def fileinput(filename):
    f = open(filename,'r')
    f.seek(0)
    l = f.readline(); #discrete or numberical
    lr = l.split(',')
    t = [int(x) for x in lr]
    lines = f.readlines()
    f.close()
    d = []
    datatag = []
    for line in lines:
        row = line.split(',')
        datarow = row[0:featureN]
        datarow = [float(x) for x in datarow]
        d.append( datarow )
        datatag.append(int(float(row[featureN])))
    #data = numpy.matrix(d)
    data = d
    return data,datatag,t



#setting 
numpy.random.seed(seed=1)  
NforNumerical = 5

"""

#dataset1
dataN = 277
featureN = 9
file = "breast-cancer-assignment5.txt"

"""

#dataset2
dataN = 1000
featureN = 24
file = "german-assignment5.txt"



#main function entry
print "fileinput...        ",
[x,y,t] = fileinput(file)
print "[done]"

name1 = -9
name2 = -9
for tag in y:
    if name1 == -9:
        name1 = tag
    else:
        if name1 != tag:
            name2 = tag
            break

"""
Tree = DT.DecisionTree(x,y,t)
print Tree.Tree

error = 0
shit = 0
for i in range(dataN):
    yy = Tree.predict(Tree.Tree,x[i],t)
    if yy != y[i]:
        error += 1
    if yy == 999:
        shit += 1
print error,dataN,shit

"""




forest = []
treeN = 10
#samplefN = int(math.log(featureN,2)) + 1
samplefN = int(featureN * 0.4)
sampledataN = int(dataN)

for i in range(treeN):
    samplex = []
    sampley = []
    samplet = []
    samplef = numpy.sort(random.sample(range(0,featureN),samplefN))
    
    for j in range(samplefN):
        samplet.append(t[samplef[j]])
    
    for j in range(sampledataN):
        rand = numpy.random.randint(0,dataN)
        xx = []
        for k in range(samplefN):
            xx.append(x[j][samplef[k]])
        samplex.append(xx)
        sampley.append(y[j])
    
    Tree = DT.DecisionTree(samplex,sampley,samplet,samplef)
    forest.append(Tree)
    #print Tree.Tree

error = 0
shit = 0
for i in range(dataN):
    count1 = 0
    count2 = 0
    for j in range(treeN):
        xx = []
        tt = []
        for k in range(len(forest[j].samplef)):
            xx.append(x[i][forest[j].samplef[k]])
            tt.append(t[forest[j].samplef[k]])
        yy = forest[j].predict(forest[j].Tree,xx,tt)
        if yy == name1:
            count1 += 1
        else:
            count2 += 1
    yy = 999
    if count1 > count2:
        yy = name1
    else:
        yy = name2
        
    if yy != y[i]:
        error += 1
    if yy == 999:
        shit += 1
print error,dataN,float(error)/dataN,shit
