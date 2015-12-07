import numpy
import math
import time
import copy
from nltk.tree import Tree
from statsmodels.sandbox.regression.try_treewalker import tree

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

def calcEntropy(y,error):
    n = len(y)
    labelCount = {}
    for label in y:
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    Entropy = 0
    
    for label in labelCount:
        p = float(labelCount[label]) / n
        if (p != 0):
            Entropy -= p * math.log(p , 2) 
    
    return Entropy

def splitSet_discrete(x,y,t,feature,Vtype):
    retx = []
    rety = []
    rett = t[:feature]
    rett.extend(t[feature+1:])
    xn = len(x)
    for i in range(xn):
        if (x[i][feature] == Vtype):
            rtx = x[i][:feature]
            rtx.extend(x[i][feature+1:])
            retx.append(rtx)
            rety.append(y[i])
    return retx,rety,rett

def splitSet_numerical(x,y,t,feature,lowerBound,upperBound):
    retx = []
    rety = []
    rett = t[:feature]
    rett.extend(t[feature+1:])
    xn = len(x)
    for i in range(xn):
        if (x[i][feature] >= lowerBound and x[i][feature] < upperBound):
            rtx = x[i][:feature]
            rtx.extend(x[i][feature+1:])
            retx.append(rtx)
            rety.append(y[i])
    return retx,rety,rett

def featureSelect(x,y,t):
    fN = len(x[0])
    initEntropy = calcEntropy(y,3)
    maxGainRatio = 0.0
    bestf = -1
    for i in range(fN):
        Entropy = 0.0
        H = 0.0
        if (t[i] == 1): #discrete
            fV = [data[i] for data in x]
            fVType = set(fV)
            for Vtype in fVType:
                [subSetx,subSety,subSett] = splitSet_discrete(x,y,t,i,Vtype)
                p = float(len(subSetx)) / float(len(x))
                Entropy += p * calcEntropy(subSety,2)
                H -= p * math.log(p,2)
        else: #numerical
            minv = 0.0
            maxv = 1.0
            delta = (maxv - minv) / NforNumerical + 0.0001
            for j in range(NforNumerical):
                [subSetx,subSety,subSett] = splitSet_numerical(x,y,t,i,minv+delta*j,minv+delta*(j+1))
                p = float(len(subSetx)) / float(len(x))
                Entropy += p * calcEntropy(subSety,1)
                if p != 0:
                    H -= p * math.log(p,2)
        if H == 0:
            H = 0.000001       
        if ((initEntropy - Entropy)/H >= maxGainRatio):
                maxGainRatio = (initEntropy - Entropy)/ H
                bestf = i
    return bestf

def classify(tags):
    tagCount = {}
    maxNum = 0
    maxTag = -9999
    for tag in tags:
        if tag not in tagCount.keys():
            tagCount[tag] = 0
        tagCount[tag] += 1
        if (tagCount[tag] > maxNum):
            maxNum = tagCount[tag]
            maxTag = tag
    return maxTag

def generateTree(x,y,t):
    tags = [tag for tag in y]
    if tags.count(tags[0]) == len(tags): # just one class
        return tags[0]
    if len(x[0]) == 0:
        return classify(tags)
    
    bestf = featureSelect(x,y,t)
    #fname = featureName[bestf]
    fname = "feature"+str(bestf)
    Tree = {fname:{}}
    

    if (t[bestf] == 1): #discrete
        fV = [data[bestf] for data in x]
        fVType = set(fV)
        for Vtype in fVType:
            [subSetx,subSety,subSett] = splitSet_discrete(x,y,t,bestf,Vtype)
            Tree[fname][Vtype] = generateTree(subSetx, subSety, subSett)
    else: #numerical
        minv = 0.0
        maxv = 1.0
        delta = (maxv - minv) / NforNumerical + 0.0001
        for i in range(NforNumerical):
            [subSetx,subSety,subSett] = splitSet_numerical(x,y,t,bestf,minv+delta*i,minv+delta*(i+1))
            if (len(subSetx) > 0):
                Tree[fname][i] = generateTree(subSetx, subSety, subSett)

    return Tree

def predict(tree,d,t):
    data = [item for item in d]
    tag = [item for item in t]
    while isinstance(tree, dict):
        fName = tree.keys()[0]
        findex = -1
        for i in range(len(data)):
            if ("feature"+str(i) == fName):
                findex = i
                break
        if tag[findex] == 1: #discrete
            try:
                tree = tree[fName][data[findex]]
            except:
                rand = numpy.random.randint(0,2)
                if rand == 0:
                    return name1
                else:
                    return name2
                return 999
        else: #numerical
            minv = 0.0
            maxv = 1.0
            delta = (maxv - minv) / NforNumerical + 0.0001
            for j in range(NforNumerical):
                if (data[findex]>=minv+delta*j and data[findex]<minv+delta*(j+1)):
                    try:
                        tree = tree[fName][j]
                    except:
                        rand = numpy.random.randint(0,2)
                        if rand == 0:
                            return name1
                        else:
                            return name2
                        return 999
                    break
        datatmp = data[:findex]
        datatmp.extend(data[findex+1:])
        data = datatmp
        tagtmp = tag[:findex]
        tagtmp.extend(tag[findex+1:])
        tag = tagtmp
    return tree

def pruning(Tree,x,y,t):
    if not isinstance(Tree, dict):
        error = 0
        right = 0
        for tag in y:
            if tag == Tree:
                right += 1
            else:
                error += 1
        return Tree,right,error
    
    nodefN = Tree.keys()[0]
    nodef = int(nodefN[7:])
    canpruning = True
    total1 = 0
    total2 = 0
    totalerror = 0
    
    newTree = {nodefN:{}}
    
    if (t[nodef] == 1): #discrete
        fV = [data[nodef] for data in x]
        fVType = set(fV)
        for Vtype in fVType:
            [subSetx,subSety,subSett] = splitSet_discrete(x,y,t,nodef,Vtype)
            try:
                [pTree, right, error ]= pruning(Tree[nodefN][Vtype],subSetx, subSety, subSett)
                newTree[nodefN][Vtype] = pTree
                if not isinstance(pTree,dict):
                    if pTree == name1:
                        total1 += right
                        total2 += error
                    else:
                        total1 += error
                        total2 += right
                    totalerror += error
                else:
                    canpruning = False
            except:
                totalerror += len(subSetx)
                if total1 < total2:
                    total1 += len(subSetx)
                else:
                    total2 += len(subSetx)    
    else: #numerical
        minv = 0.0
        maxv = 1.0
        delta = (maxv - minv) / NforNumerical + 0.0001
        for i in range(NforNumerical):
            [subSetx,subSety,subSett] = splitSet_numerical(x,y,t,nodef,minv+delta*i,minv+delta*(i+1))
            if (len(subSetx) > 0):
                try:
                    [pTree, right, error ]= pruning(Tree[nodefN][i],subSetx, subSety, subSett)
                    newTree[nodefN][i] = pTree
                    if not isinstance(pTree,dict):
                        if pTree == name1:
                            total1 += right
                            total2 += error
                        else:
                            total1 += error
                            total2 += right
                        totalerror += error
                    else:
                        canpruning = False
                except:
                    totalerror += len(subSetx)
                    if total1 < total2:
                        total1 += len(subSetx)
                    else:
                        total2 += len(subSetx) 

    if canpruning and totalerror > min(total1,total2):
        #print name1,name2,totalerror,total1,total2
        if total1 < total2:
            #print Tree,totalerror,total1,total2,name1
            return name2,total2,total1
        else:
            #print Tree,totalerror,total1,total2,name2
            return name1,total1,total2
        
    return newTree,len(x)-totalerror,totalerror

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
NforNumerical = 10

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
#[tx,ty] = fileinput(file+"testing.txt")
print "[done]"

#t = analysisdata(x,y,t)

name1 = -9
name2 = -9
for tag in y:
    if name1 == -9:
        name1 = tag
    else:
        if name1 != tag:
            name2 = tag
            break


xsplit = []
ysplit = []
for i in range(10):
    xsplit.append( x[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )
    ysplit.append( y[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )


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
    
    
    #pruning, 80% for train, 20% for pruning
    """
    p80 = 1 - len(x) / 4
    trainx = x[:p80]
    trainy = y[:p80]
    validx = x[p80:]
    validy = y[p80:]
    Tree = generateTree(trainx,trainy,t)
    [Tree,aa,bb] = pruning(Tree,validx,validy,t)
    """
    
    #without pruning
    Tree = generateTree(x, y, t)
    
    error = 0
    for i in range(len(validationx)):
        yy = predict(Tree,validationx[i], t)
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













"""
p80 = 1 - dataN / 4
trainx = x[:p80]
trainy = y[:p80]
validx = x[p80:]
validy = y[p80:]

Tree = generateTree(trainx,trainy,t)
print Tree


[pTree,aa,bb] = pruning(Tree,validx,validy,t)

print pTree

error = 0
shit = 0
for i in range(dataN):
    yy = predict(Tree,x[i])
    if yy != y[i]:
        error += 1
    if yy == 999:
        shit += 1
print "unpruning on full set: ",error,dataN,shit

error = 0
shit = 0
for i in range(dataN):
    yy = predict(pTree,x[i])
    if yy != y[i]:
        error += 1
    if yy == 999:
        shit += 1
print "pruning on full set: ",error,dataN,shit

error = 0
shit = 0
validxn = len(validx)
for i in range(validxn):
    yy = predict(Tree,validx[i])
    if yy != validy[i]:
        error += 1
    if yy == 999:
        shit += 1
print "unpruning on validition set: ",error,validxn,shit

error = 0
shit = 0
for i in range(validxn):
    yy = predict(pTree,validx[i])
    if yy != validy[i]:
        error += 1
    if yy == 999:
        shit += 1
print "pruning on validition set: ",error,validxn,shit
    
"""