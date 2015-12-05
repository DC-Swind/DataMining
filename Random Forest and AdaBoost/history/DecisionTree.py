import numpy
import math
import time
import copy
from statsmodels.sandbox.regression.try_treewalker import tree

class DecisionTree:
    
    def __init__(self,x,y,t,f):
        numpy.random.seed(seed=1)  
        self.NforNumerical = 5
        self.samplef = f
        self.featureN = len(t)
        
        self.name1 = -9
        self.name2 = -9
        for tag in y:
            if self.name1 == -9:
                self.name1 = tag
            else:
                if self.name1 != tag:
                    self.name2 = tag
                    break
        self.Tree = self.generateTree(x, y, t)
    
    def fileinput(self,filename):
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
            datarow = row[0:self.featureN]
            datarow = [float(x) for x in datarow]
            d.append( datarow )
            datatag.append(int(float(row[self.featureN])))
        #data = numpy.matrix(d)
        data = d
        return data,datatag,t

    def calcEntropy(self,y,error):
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

    def splitSet_discrete(self,x,y,t,feature,Vtype):
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

    def splitSet_numerical(self,x,y,t,feature,lowerBound,upperBound):
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

    def featureSelect(self,x,y,t):
        fN = len(x[0])
        initEntropy = self.calcEntropy(y,3)
        maxGain = 0.0
        bestf = -1
        for i in range(fN):
            if (t[i] == 1): #discrete
                fV = [data[i] for data in x]
                fVType = set(fV)
                Entropy = 0.0
                for Vtype in fVType:
                    [subSetx,subSety,subSett] = self.splitSet_discrete(x,y,t,i,Vtype)
                    p = float(len(subSetx)) / float(len(x))
                    Entropy += p * self.calcEntropy(subSety,2)
            else: #numerical
                minv = 0.0
                maxv = 1.0
                Entropy = 0.0
                delta = (maxv - minv) / self.NforNumerical + 0.0001
                for j in range(self.NforNumerical):
                    [subSetx,subSety,subSett] = self.splitSet_numerical(x,y,t,i,minv+delta*j,minv+delta*(j+1))
                    p = float(len(subSetx)) / float(len(x))
                    Entropy += p * self.calcEntropy(subSety,1)       
            if (initEntropy - Entropy >= maxGain):
                maxGain = initEntropy - Entropy
                bestf = i
        return bestf

    def classify(self,tags):
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

    def generateTree(self,x,y,t):
        tags = [tag for tag in y]
        if tags.count(tags[0]) == len(tags): # just one class
            return tags[0]
        if len(x[0]) == 0:
            return self.classify(tags)
    
        bestf = self.featureSelect(x,y,t)
        #fname = featureName[bestf]
        fname = "feature"+str(bestf)
        Tree = {fname:{}}
    

        if (t[bestf] == 1): #discrete
            fV = [data[bestf] for data in x]
            fVType = set(fV)
            for Vtype in fVType:
                [subSetx,subSety,subSett] = self.splitSet_discrete(x,y,t,bestf,Vtype)
                Tree[fname][Vtype] = self.generateTree(subSetx, subSety, subSett)
        else: #numerical
            minv = 0.0
            maxv = 1.0
            delta = (maxv - minv) / self.NforNumerical + 0.0001
            for i in range(self.NforNumerical):
                [subSetx,subSety,subSett] = self.splitSet_numerical(x,y,t,bestf,minv+delta*i,minv+delta*(i+1))
                if (len(subSetx) > 0):
                    Tree[fname][i] = self.generateTree(subSetx, subSety, subSett)

        return Tree

    def predict(self,tree,d,t):
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
                        return self.name1
                    else:
                        return self.name2
                    return 999
            else: #numerical
                minv = 0.0
                maxv = 1.0
                delta = (maxv - minv) / self.NforNumerical + 0.0001
                for j in range(self.NforNumerical):
                    if (data[findex]>=minv+delta*j and data[findex]<minv+delta*(j+1)):
                        try:
                            tree = tree[fName][j]
                        except:
                            rand = numpy.random.randint(0,2)
                            if rand == 0:
                                return self.name1
                            else:
                                return self.name2
                            return 999
                        break
            datatmp = data[:findex]
            datatmp.extend(data[findex+1:])
            data = datatmp
            tagtmp = tag[:findex]
            tagtmp.extend(tag[findex+1:])
            tag = tagtmp
        return tree

"""

error = 0
shit = 0
for i in range(dataN):
    yy = predict(Tree,x[i])
    if yy != y[i]:
        error += 1
    if yy == 999:
        shit += 1
print error,dataN,shit

"""