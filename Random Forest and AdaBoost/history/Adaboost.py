import numpy

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

def train(x,y,t,M):
    G = {}
    alpha = {}
    for i in range(M):
        G.setdefault(i)
        alpha.setdefault(i)
    for i in range(M):
        G[i] = DecisionTree(x,y,t)

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

train(x,y,t,4)
