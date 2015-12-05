import numpy
import time
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


xsplit = []
ysplit = []
for i in range(10):
    xsplit.append( totalx[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )
    ysplit.append( totaly[(int)(dataN*i/10):(int)(dataN*(i+1)/10)] )


forest = []
treeN = 10
#samplefN = int(math.log(featureN,2)) + 1
#samplefN = int(featureN * 0.4)
sampledataN = int(dataN * 0.9)
totalerror = 0
totalvaladition = 0
shit = 0
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
    for i in range(treeN):
        samplex = []
        sampley = []
    
        for j in range(sampledataN):
            rand = numpy.random.randint(0,sampledataN)
            samplex.append(x[rand])
            sampley.append(y[rand])
    
        Tree = DT.DecisionTree(samplex,sampley,t)
        forest.append(Tree)
        #print Tree.Tree

    error = 0
    for i in range(len(validationx)):
        count1 = 0
        count2 = 0
        for j in range(treeN):
            yy = forest[j].predict(forest[j].Tree,validationx[i],t)
            if yy == 999:
                continue
            if yy == name1:
                count1 += 1
            else:
                count2 += 1

        if count1 + count2 == 0:
            shit += 1
        yy = 999
        if count1 > count2:
            yy = name1
        else:
            yy = name2
        
        if yy != validationy[i]:
            error += 1
    #print error,len(validationx),shit
    totalerror += error
    totalvaladition += len(validationx)

print totalerror,totalvaladition,totalerror/float(totalvaladition),shit