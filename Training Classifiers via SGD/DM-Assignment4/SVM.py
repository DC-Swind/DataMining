import numpy
import matplotlib.pyplot as plt

def fileinput(filename):
    f = open(filename,'r')
    f.seek(0)
    lines = f.readlines()
    f.close()
    d = []
    datatag = []
    for line in lines:
        row = line.split(',')
        datarow = row[0:featureN]
        datarow = [int(x) for x in datarow]
        d.append( datarow )
        datatag.append(int(row[featureN]))
    data = numpy.matrix(d)
    return data,datatag

def test(w):
    right = 0
    error = 0
    for i in range(0,tdataN):
        z = numpy.inner(tx[i],w)
        if (z * ty[i] > 0):
            right = right + 1
        else:
            error = error + 1
    return float(error)/float(tdataN)

def pegasos_hinge(S,lamda,T):
    figurey = numpy.zeros(10)
    figurex = numpy.zeros(10)
    x = S
    index = 1
    w = numpy.zeros(featureN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)
        
        if (y[i] * numpy.inner(w,x[i]) < 1):
            w = ( 1 - yita * lamda ) * w + yita * y[i] * x[i]
        else:
            w = ( 1 - yita * lamda ) * w
            
        if (int(float(T * index) / 10) == t and figurey[index-1] == 0):
            figurex[index-1] = t
            figurey[index-1] = test(w)
            index = index + 1
    return w,figurex,figurey

def pegasos_log(S,lamda,T):
    figurey = numpy.zeros(10)
    figurex = numpy.zeros(10)
    index = 1
    x = S
    w = numpy.zeros(featureN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)
        z = numpy.inner(w,x[i])
        if (y[i] * z > 100):
            expp = 100
        else:
            expp = y[i] * z
        w = ( 1 - yita * lamda ) * w + yita * (y[i] / (1 + numpy.exp(expp))) * x[i]
        
        if (int(float(T * index) / 10) == t and figurey[index-1] == 0):
            figurex[index-1] = t
            figurey[index-1] = test(w)
            index = index + 1
            
    return w,figurex,figurey

#setting 
numpy.random.seed(seed=1)  

"""  
#dataset1
dataN = 22696
featureN = 123
tdataN = 9865
lamda = 0..0001
file = "dataset1-a8a-"
"""

#dataset2
dataN = 32561
featureN = 123
tdataN = 16281
file = "dataset1-a9a-"
lamda = 0.00005


#main function entry
print "fileinput...        ",
[x,y] = fileinput(file+"training.txt")
[tx,ty] = fileinput(file+"testing.txt")
print "[done]"

print "training...        ",
#[w,figurex,figurey] = pegasos_hinge(x, lamda, 5 * dataN)
[w,figurex,figurey] = pegasos_log(x,lamda, 5 * dataN)
print "[done]"

arix = [0.1] * 10
for i in range(1,11):
    arix[i-1] = arix[i-1] * i
plt.plot(arix,figurey)
plt.xlabel('iterater times(percent of T)')
plt.ylabel('error rate')
plt.ylim(0,1)
plt.show()
print "program is done."
print "figure-y: ",figurey