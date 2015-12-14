from preprocess import readfile
import numpy

def pegasos_log(x,y,lamda,T):
    w = numpy.zeros(featureN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)
        z = numpy.inner(w,x[i])
        expp = min(y[i] * z,100)
        w = ( 1 - yita * lamda ) * w + yita * (y[i] / (1 + numpy.exp(expp))) * x[i]    
    return w

def pegasos_hinge(x,y,lamda,T):
    w = numpy.zeros(featureN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)

        if (y[i] * numpy.inner(w,x[i]) < 1):
            w = ( 1 - yita * lamda ) * w + yita * y[i] * x[i]
        else:
            w = ( 1 - yita * lamda ) * w
        
    return w

def K(x1,x2):
    return numpy.exp(-numpy.inner((x1-x2),(x1-x2))/(2*sigma2))

def pegasos_kernel(x,y,lamda,T):
    w = numpy.zeros(dataN)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        yita = 1 / (lamda * t)

        yy = 0
        for j in range(dataN):
            yy += w[j] * y[j] * K(x[j],x[i])
        yy *= y[i] * yita
        if yy < 1:
            w[i] = w[i] + 1
    return w

def kerneltest(x,y,w):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    dataN = len(x)
    for i in range(dataN):
        z = 0
        for j in range(dataN):
            z += w[j]*y[j]*K(x[j],x[i])
        if (z * y[i] > 0):
            right = right + 1
            if y[i] == 1:
                callright += 1
        else:
            error = error + 1
            if y[i] == 1:
                callerror += 1
    return right,error,callright,callerror


def test(x,y,w):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    dataN = len(x)
    for i in range(dataN):
        z = numpy.inner(x[i],w)
        if (z * y[i] > 0):
            right = right + 1
            if y[i] == 1:
                callright += 1
        else:
            error = error + 1
            if y[i] == 1:
                callerror += 1
    return right,error,callright,callerror

#main entry
numpy.random.seed(seed=1) 
x,y = readfile("train.csv")
dataN = len(x)
featureN = len(x[0].transpose())
print dataN,featureN
lamda = 0.00005
sigma2 = 1
for i in range(8):
    print "start SVM",i+1,"    ",
    yy = []
    for t in y:
        if t == i+1:
            yy.append(1)
        else:
            yy.append(-1)
    #w = pegasos_log(x,yy,lamda, 5 * dataN)
    #w = pegasos_hinge(x,yy,lamda, 5 * dataN)
    #right,error,callright,callerror = test(x,yy,w)
    
    w = pegasos_kernel(x,yy,lamda, 5 * dataN)
    right,error,callright,callerror = kerneltest(x,yy,w)
    print right,error,callright,callerror
