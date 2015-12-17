import csv
import os
import time
import numpy


<<<<<<< HEAD
def checkfeature(data,fname):
    rt = []
    for featurei in range(len(data[0])):
        ftype = type(data[0][featurei])
        #print featurei,fname[featurei],ftype,
        if ftype == type(1.1) or ftype == type(1):
            maxv = 0.0
            minv = 9999.0
            for d in data:
                if d[featurei] > maxv:
                    maxv = d[featurei]
                if d[featurei] < minv:
                    minv = d[featurei]
            #print maxv,minv
            rt.append(maxv)
        if ftype == type("str"):
            v = []
            for d in data:
                v.append(d[featurei])
            vset = set(v)
            #print len(vset)
            rt.append(0)
    return rt
=======
def checkfeature(data,fname,featurei):
    ftype = type(data[0][featurei])
    print fname[featurei],ftype,
    if ftype == type(1):
        maxv = 0
        minv = 9999
        for d in data:
            if d[featurei] > maxv:
                maxv = d[featurei]
            if d[featurei] < minv:
                minv = d[featurei]
        print maxv,minv
    if ftype == type(1.1):
        maxv = 0.0
        minv = 9999.0
        for d in data:
            if d[featurei] > maxv:
                maxv = d[featurei]
            if d[featurei] < minv:
                minv = d[featurei]
        print maxv,minv
    if ftype == type("str"):
        v = []
        for d in data:
            v.append(d[featurei])
        vset = set(v)
        print len(vset)
    return 0
>>>>>>> origin/master
def readfile(filename):
    discreteSet = ["Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
    for i in range(48):
        discreteSet.append("Medical_Keyword_"+str(i+1))
    discreteSet.extend(["Product_Info_1","Product_Info_3","Product_Info_5","Product_Info_6","Product_Info_7","Employment_Info_2","Employment_Info_3","Employment_Info_5","InsuredInfo_1","InsuredInfo_2","InsuredInfo_3","InsuredInfo_4","InsuredInfo_5","InsuredInfo_6","InsuredInfo_7","Insurance_History_1","Insurance_History_2","Insurance_History_3","Insurance_History_4","Insurance_History_7","Insurance_History_8","Insurance_History_9","Family_Hist_1","Medical_History_3","Medical_History_4","Medical_History_5","Medical_History_6","Medical_History_7","Medical_History_8","Medical_History_9","Medical_History_11","Medical_History_12","Medical_History_13","Medical_History_14","Medical_History_16","Medical_History_17","Medical_History_18","Medical_History_19","Medical_History_20","Medical_History_21","Medical_History_22","Medical_History_23","Medical_History_25","Medical_History_26","Medical_History_27","Medical_History_28","Medical_History_29","Medical_History_30","Medical_History_31","Medical_History_33","Medical_History_34","Medical_History_35","Medical_History_36","Medical_History_37","Medical_History_38","Medical_History_39","Medical_History_40","Medical_History_41"])
    continuesSet = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
    continuesSet.extend(["Medical_History_2","Medical_History_10"])
    
<<<<<<< HEAD
    missvalue = ['Family_Hist_3', 'Family_Hist_2', 'Medical_History_15', 'Family_Hist_5', 'Medical_History_10', 'Family_Hist_4', 'Medical_History_32', 'Medical_History_24', 'Medical_History_1', 'Employment_Info_4', 'Employment_Info_6', 'Employment_Info_1', 'Insurance_History_5']
    
=======
>>>>>>> origin/master
    print "start reading file:",filename
    csvfile = file(os.path.join(os.getcwd(), filename),"rb")
    reader = csv.reader(csvfile)
    data = []
    dataN = 0
    featureN = 0
    fname = []
<<<<<<< HEAD
    set = {}
=======
>>>>>>> origin/master
    for line in reader:
        if line[0] == "Id":
            fname.extend( line )
            continue
        dataN += 1
        if featureN == 0:
            featureN = len(line)
        d = []
        for i in range(featureN):
<<<<<<< HEAD
            """
            if fname[i] in missvalue:
                d.append(0)
                continue
            """
            if fname[i] in discreteSet:
                if line[i] == "":
                    d.append(-1)
                    set[fname[i]] = 1
=======
            if fname[i] in discreteSet:
                if line[i] == "":
                    d.append(-1)
>>>>>>> origin/master
                    continue
                d.append(int(line[i]))
                continue
            if fname[i] in continuesSet:
                if line[i] == "":
                    d.append(-1.0)
<<<<<<< HEAD
                    set[fname[i]] = 1
=======
>>>>>>> origin/master
                    continue
                try:
                    d.append(float(line[i]))
                    continue
                except:
                    d.append(float(int(line[i])))
                    continue
            if i == 2:
                d.append(int(line[i],16))
                continue
            d.append(line[i])
        data.append(d)
    #print "dataN:",dataN," featureN:",featureN
<<<<<<< HEAD
    print set
    
    maxv = checkfeature(data,fname)
    dataN = len(data)
    AA = numpy.zeros((dataN,featureN-2))
    t = []
    index = 0
    for d in data:
        dd = numpy.zeros(featureN-2)
        for i in range(featureN-2):
            dd[i] = float(d[i+1])/maxv[i+1]
            #dd.append(float(d[i+1]))
        AA[index,:] = dd
        t.append(int(d[featureN-1]))
        index += 1
    #checkfeature(AA,fname)
    #A = numpy.matrix(AA)
    csvfile.close()
    return AA,t



def readtestfile(filename):
    discreteSet = ["Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
    for i in range(48):
        discreteSet.append("Medical_Keyword_"+str(i+1))
    discreteSet.extend(["Product_Info_1","Product_Info_3","Product_Info_5","Product_Info_6","Product_Info_7","Employment_Info_2","Employment_Info_3","Employment_Info_5","InsuredInfo_1","InsuredInfo_2","InsuredInfo_3","InsuredInfo_4","InsuredInfo_5","InsuredInfo_6","InsuredInfo_7","Insurance_History_1","Insurance_History_2","Insurance_History_3","Insurance_History_4","Insurance_History_7","Insurance_History_8","Insurance_History_9","Family_Hist_1","Medical_History_3","Medical_History_4","Medical_History_5","Medical_History_6","Medical_History_7","Medical_History_8","Medical_History_9","Medical_History_11","Medical_History_12","Medical_History_13","Medical_History_14","Medical_History_16","Medical_History_17","Medical_History_18","Medical_History_19","Medical_History_20","Medical_History_21","Medical_History_22","Medical_History_23","Medical_History_25","Medical_History_26","Medical_History_27","Medical_History_28","Medical_History_29","Medical_History_30","Medical_History_31","Medical_History_33","Medical_History_34","Medical_History_35","Medical_History_36","Medical_History_37","Medical_History_38","Medical_History_39","Medical_History_40","Medical_History_41"])
    continuesSet = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
    continuesSet.extend(["Medical_History_2","Medical_History_10"])
    
    print "start reading file:",filename
    csvfile = file(os.path.join(os.getcwd(), filename),"rb")
    reader = csv.reader(csvfile)
    data = []
    dataN = 0
    featureN = 0
    fname = []
    for line in reader:
        if line[0] == "Id":
            fname.extend( line )
            continue
        dataN += 1
        if featureN == 0:
            featureN = len(line)
        d = []
        for i in range(featureN):
            if fname[i] in discreteSet:
                if line[i] == "":
                    d.append(-1)
                    continue
                d.append(int(line[i]))
                continue
            if fname[i] in continuesSet:
                if line[i] == "":
                    d.append(-1.0)
                    continue
                try:
                    d.append(float(line[i]))
                    continue
                except:
                    d.append(float(int(line[i])))
                    continue
            if i == 2:
                d.append(int(line[i],16))
                continue
            d.append(line[i])
        data.append(d)
    #print "dataN:",dataN," featureN:",featureN
    
    #for i in range(featureN):
    #    checkfeature(data,fname,i)
    
    maxv = checkfeature(data,fname)
    dataN = len(data)
    AA = numpy.zeros((dataN,featureN-1))
    ID = []
    index = 0
    for d in data:
        dd = numpy.zeros(featureN-1)
        for i in range(featureN-1):
            dd[i] = float(d[i+1])/maxv[i+1]
            #dd.append(float(d[i+1]))
        AA[index,:] = dd
        ID.append(int(d[0]))
        index += 1
        
    csvfile.close()
    return ID,AA



=======
    
    #for i in range(featureN):
    #    checkfeature(data,fname,i)
    
    AA = []
    t = []
    for d in data:
        dd = []
        for i in range(featureN-2):
            dd.append(float(d[i+1]))
        AA.append(dd)
        t.append(int(d[featureN-1]))
    A = numpy.matrix(AA)
    
    return A,t
>>>>>>> origin/master
# main entry
"""
a = [1,2,3,4,5,6,7,8,9]
dd = []
for i in range(8):
    dd.append(float(a[i+1]))
print dd
"""

#x,y = readfile("train.csv")
  
