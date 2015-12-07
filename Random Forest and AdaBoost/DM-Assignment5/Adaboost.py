import numpy
import time
import math
import random
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

def wRandom(w):
    l = len(w)
    wsum = 0
    rand = random.random()
    rt = -1
    for i in range(l-1):
        wsum = wsum + w[i]
        if wsum > rand:
            rt = i
            break 
    if rt == -1:
        rt = l - 1
    return rt

def train(x,y,t,T):
    G = {}  # weak classifier
    alpha = {}   #talk weight
    dataN = len(x)  #random or not
    W = [1.0/dataN] * dataN
    for i in range(T):
        G.setdefault(i)
        alpha.setdefault(i)
        
    for i in range(T):
        samplex = []
        sampley = []
        for j in range(dataN):
            #rand = numpy.random.randint(0,dataN)
            rand = wRandom(W)
            samplex.append(x[rand])
            sampley.append(y[rand])
    
        G[i] = DT.DecisionTree(samplex,sampley,t)
        e = 0
        yy = {}
        counterror = 0
        for j in range(dataN):
            yy.setdefault(j)
            yy[j] = G[i].predict(G[i].Tree,x[j],t)
            if yy[j] != y[j]:
                e += W[j]
                counterror += 1
        #print counterror, dataN , counterror/float(dataN)
        alpha[i] = math.log((1-e)/e,numpy.e)/2
        
        Z = 0
        for j in range(dataN):
            if yy[j] == y[j]:
                Z += W[j] * math.exp(-alpha[i])
            else:
                Z += W[j] * math.exp(alpha[i])
        for j in range(dataN):
            if yy[j] == y[j]:
                W[j] = W[j] * math.exp(-alpha[i]) / Z
            else:
                W[j] = W[j] * math.exp(alpha[i]) / Z
    alphasum = 0
    for i in range(WeakerN):
        alphasum += alpha[i]
    for i in range(WeakerN):
        alpha[i] = alpha[i]/alphasum
    return G,alpha   

def predict(G,alpha,x,t):
    yy = 0
    for j in range(WeakerN):
        yj = G[j].predict(G[j].Tree,x,t)
        if yj == 999:
            continue
        else:
            yy += alpha[j] * yj
    
    if yy > (name1 + name2)/2.0:
        yy = name2
    else:
        yy = name1
    return yy


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


WeakerN = 20

#main function entry
print "fileinput...        ",
[totalx,totaly,t] = fileinput(file)
print "[done]"

name1 = -9
name2 = -9
for tag in totaly:
    if name1 == -9:
        name1 = tag
    else:
        if name1 != tag:
            name2 = tag
            break
if name1 > name2:
    tmp = name1
    name1 = name2
    name2 = tmp

xsplit = []
ysplit = []

for i in range(10):
    xsplit.append( totalx[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )
    ysplit.append( totaly[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )
"""
for i in range(10):
    xsplit.append([])
    ysplit.append([])
for i in range(dataN):
    xsplit[i%10].append(totalx[i])
    ysplit[i%10].append(totaly[i])
"""


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
            
    G,alpha = train(x,y,t,WeakerN)
    
    error = 0
    for i in range(len(validationx)):
        yy = predict(G, alpha, validationx[i], t)
        if yy != validationy[i]:
            error += 1
    Accuray[cross] = 1 - float(error) / len(validationx)
    
for i in range(10):
    mean += Accuray[i]
mean = mean / 10
for i in range(10):
    StandardDeviation += (Accuray[i] - mean) * (Accuray[i] - mean)
StandardDeviation = math.sqrt(StandardDeviation / 10)
print "mean: ",mean,"StandardDeviation: ",StandardDeviation