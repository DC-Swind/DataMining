from preprocess import readfile
import numpy


def H(w,x):
    print x
    x = x.extend(1)
    return 1/(1 + numpy.exp(-w.tranpose * x))

def pegasos_lr(x,y,lamda,T):
    w = numpy.zeros(featureN+1)
    for t in range(1,T+1):
        i = numpy.random.randint(0,dataN)
        w = w - lamda * (H(w,x[i])-y[i])*x[i]
    return w


def test(x,y,w):
    right = 0
    error = 0
    callright = 0
    callerror = 0
    dataN = len(x)
    for i in range(dataN):
        z = H(w, x[i])
        if (z * y[i] >= 0.5):
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

for i in range(8):
    print "start LR",i+1,"    ",
    yy = []
    for t in y:
        if t == i+1:
            yy.append(1)
        else:
            yy.append(0)
    
    w = pegasos_lr(x,yy,lamda, 5 * dataN)
    right,error,callright,callerror = test(x,yy,w)
<<<<<<< HEAD
    print right,error,callright,callerror
=======
    print right,error,callright,callerror
>>>>>>> origin/master
