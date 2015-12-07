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

def calcEntropy(y):
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

def splitSet_numerical(x,y,t,feature,splitdot):
    ret1x = []
    ret1y = []
    ret2x = []
    ret2y = []
    rett = t[:feature]
    rett.extend(t[feature+1:])
    xn = len(x)
    for i in range(xn):
        if (x[i][feature] <= splitdot):
            rtx = x[i][:feature]
            rtx.extend(x[i][feature+1:])
            ret1x.append(rtx)
            ret1y.append(y[i])
        else:
            rtx = x[i][:feature]
            rtx.extend(x[i][feature+1:])
            ret2x.append(rtx)
            ret2y.append(y[i])
    return ret1x,ret1y,ret2x,ret2y,rett

def featureSelect(x,y,t):
    fN = len(x[0])
    initEntropy = calcEntropy(y)
    maxGainRatio = 0.0
    bestf = -1
    bestV = -999
    for i in range(fN):
        Entropy = 0.0
        H = 0.0
        bestVtype = -999
        if (t[i] == 1): #discrete
            fV = [data[i] for data in x]
            fVType = set(fV)
            for Vtype in fVType:
                [subSetx,subSety,subSett] = splitSet_discrete(x,y,t,i,Vtype)
                p = float(len(subSetx)) / float(len(x))
                Entropy += p * calcEntropy(subSety)
                H -= p * math.log(p,2)
        else: #numerical
            fV = [data[i] for data in x]
            fVType = set(fV)
            maxGain = 0.0
            for Vtype in fVType:
                [subset1x,subset1y,subset2x,subset2y,subsett] = splitSet_numerical(x, y, t, i,Vtype)
                p1 = float(len(subset1x))/len(x)
                p2 = float(len(subset2x))/len(x)
                Entropyt = p1 * calcEntropy(subset1y) + p2 * calcEntropy(subset2y)
                if (initEntropy - Entropyt)>maxGain:
                    maxGain = initEntropy - Entropyt
                    bestVtype = Vtype
            
            [subset1x,subset1y,subset2x,subset2y,subsett] = splitSet_numerical(x, y, t, i,bestVtype)
            p1 = float(len(subset1x))/len(x)
            p2 = float(len(subset2x))/len(x)
            Entropy = p1 * calcEntropy(subset1y) + p2 * calcEntropy(subset2y)
            if p1 != 0:
                H -= p1 * math.log(p1,2)
            if p2 != 0:
                H -= p2 * math.log(p2,2)
        if H == 0:
            H = 0.000001       
        if ((initEntropy - Entropy)/H >= maxGainRatio):
            maxGainRatio = (initEntropy - Entropy)/ H
            bestf = i
            bestV = bestVtype
    return bestf,bestVtype

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
    
    bestf,bestv = featureSelect(x,y,t)
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
        [subSet1x,subSet1y,subSet2x,subSet2y,subSett] = splitSet_numerical(x,y,t,bestf,bestv)
        if (len(subSet1x) > 0):
            Tree[fname][0] = generateTree(subSet1x, subSet1y, subSett)
        if (len(subSet2x) > 0):
            Tree[fname][1] = generateTree(subSet2x, subSet2y, subSett)
        Tree[fname][3] = bestv
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
            try:
                if (data[findex]<=tree[fName][3]):
                    tree = tree[fName][0]
                else:
                    tree = tree[fName][1]
            except:
                rand = numpy.random.randint(0,2)
                if rand == 0:
                    return name1
                else:
                    return name2
                return 999

        datatmp = data[:findex]
        datatmp.extend(data[findex+1:])
        data = datatmp
        tagtmp = tag[:findex]
        tagtmp.extend(tag[findex+1:])
        tag = tagtmp
    return tree



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

t = analysisdata(x,y,t)

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
