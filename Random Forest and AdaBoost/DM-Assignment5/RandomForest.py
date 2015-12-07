import numpy
import time
import math
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

def train(x,y,t):
    Forest = []
    sampledataN = len(y)
    for i in range(treeN):
        samplex = []
        sampley = []
        for j in range(sampledataN):
            rand = numpy.random.randint(0,sampledataN)
            samplex.append(x[rand])
            sampley.append(y[rand])
    
        Tree = DT.DecisionTree(samplex,sampley,t)
        Forest.append(Tree)
        #print Tree.Tree
    return Forest

def predict(forest,x,t):
    count1 = 0
    count2 = 0
    for j in range(treeN):
        yy = forest[j].predict(forest[j].Tree,x,t)
        if yy == 999:
            continue
        if yy == name1:
            count1 += 1
        else:
            count2 += 1
    yy = 999
    if count1 > count2:
        yy = name1
    else:
        yy = name2
        
    return yy

def analysisdata(x,y,t):
    for i in range(featureN):
        if t[i] == 0:
            S = {}
            for data in x:
                S.setdefault(data[i])
            if len(S) <= 5:
                t[i] = 1
    return t

 
#setting 
numpy.random.seed(seed=1)  
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


treeN = 20


#main function entry
print "fileinput...        ",
[totalx,totaly,t] = fileinput(file)
print "[done]"

#t = analysisdata(totalx, totaly, t)

for i in range(featureN):
    if t[i] == 0:
        S = {}
        for data in totalx:
            S.setdefault(data[i])
        if len(S) <=5:
            t[i] = 1
    


name1 = -9
name2 = -9
for tag in totaly:
    if name1 == -9:
        name1 = tag
    else:
        if name1 != tag:
            name2 = tag
            break


xsplit = []
ysplit = []
for i in range(10):
    xsplit.append( totalx[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )
    ysplit.append( totaly[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )



mean = 0
StandardDeviation = 0
Accuray = [0.0] * 10

for cross in range(10):
    x = []
    y = []
    validationx = []
    validationy = []
    for i in range(10):
        if i == cross:
            validationx.extend(xsplit[i])
            validationy.extend(ysplit[i])
        else:
            x.extend(xsplit[i])
            y.extend(ysplit[i])
    
    forest = train(x, y, t)
    
    error = 0
    for i in range(len(validationx)):
        yy = predict(forest, validationx[i], t)
        if yy != validationy[i]:
            error += 1
    #print error,len(validationx),shit
    Accuray[cross] = 1 - float(error) / len(validationx)

for i in range(10):
    mean += Accuray[i]
mean = mean / 10
for i in range(10):
    StandardDeviation += (Accuray[i] - mean) * (Accuray[i] - mean)
StandardDeviation = math.sqrt(StandardDeviation / 10)
print "mean: ",mean,"StandardDeviation: ",StandardDeviation