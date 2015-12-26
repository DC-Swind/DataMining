import csv
import os
import time
import numpy

class process():
    def __init__(self):
        self.discreteSet = ["Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
        for i in range(48):
            self.discreteSet.append("Medical_Keyword_"+str(i+1))
        self.discreteSet.extend(["Product_Info_1","Product_Info_3","Product_Info_5","Product_Info_6","Product_Info_7","Employment_Info_2","Employment_Info_3","Employment_Info_5","InsuredInfo_1","InsuredInfo_2","InsuredInfo_3","InsuredInfo_4","InsuredInfo_5","InsuredInfo_6","InsuredInfo_7","Insurance_History_1","Insurance_History_2","Insurance_History_3","Insurance_History_4","Insurance_History_7","Insurance_History_8","Insurance_History_9","Family_Hist_1","Medical_History_3","Medical_History_4","Medical_History_5","Medical_History_6","Medical_History_7","Medical_History_8","Medical_History_9","Medical_History_11","Medical_History_12","Medical_History_13","Medical_History_14","Medical_History_16","Medical_History_17","Medical_History_18","Medical_History_19","Medical_History_20","Medical_History_21","Medical_History_22","Medical_History_23","Medical_History_25","Medical_History_26","Medical_History_27","Medical_History_28","Medical_History_29","Medical_History_30","Medical_History_31","Medical_History_33","Medical_History_34","Medical_History_35","Medical_History_36","Medical_History_37","Medical_History_38","Medical_History_39","Medical_History_40","Medical_History_41"])
        self.continuesSet = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
        self.continuesSet.extend(["Medical_History_2","Medical_History_10"])
        #missvalue = ['Family_Hist_3', 'Family_Hist_2', 'Medical_History_15', 'Family_Hist_5', 'Medical_History_10', 'Family_Hist_4', 'Medical_History_32', 'Medical_History_24', 'Medical_History_1', 'Employment_Info_4', 'Employment_Info_6', 'Employment_Info_1', 'Insurance_History_5']
   
        self.maxv = []
        self.minv = []
        
    def analysis_data(self, data, fname):
        maxv = []
        minv = []
        for i in range(1, len(fname) - 1):
            minvv = 99999999.9
            maxvv = -9999999.9
            for d in data:
                if float(d[i]) > maxvv:
                    maxvv = float(d[i])
                if float(d[i]) > -1 and float(d[i]) < minvv:
                    minvv = float(d[i])
            maxv.append(maxvv)
            minv.append(minvv)
        return maxv, minv

    def readtrainfile(self, filename):
        cputime = time.time()
        print "start reading training file:",filename,"    ",
        csvfile = file(os.path.join(os.getcwd(), filename),"rb")
        reader = csv.reader(csvfile)
        data = []
        featureN = 0
        fname = []
        missvset = {}
        for line in reader:
            if line[0] == "Id":
                fname.extend( line )
                continue

            if featureN == 0:
                featureN = len(line)
            d = []
            
            for i in range(featureN):
                if fname[i] in self.discreteSet:
                    if line[i] == "":
                        d.append(-1)
                        missvset[fname[i]] = 1
                        continue
                    d.append(int(line[i]))
                    continue
                if fname[i] in self.continuesSet:
                    if line[i] == "":
                        d.append(-1.0)
                        missvset[fname[i]] = 1
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
        
        self.maxv, self.minv = self.analysis_data(data,fname)
        #print missvset
    
        dataN = len(data)
        AA = numpy.zeros((dataN,featureN-2))
        t = []
        index = 0
        for d in data:
            dd = numpy.zeros(featureN-2)
            for i in range(featureN-2):
                dd[i] = (float(d[i+1]) - self.minv[i])/self.maxv[i]
            AA[index,:] = dd
            t.append(int(d[featureN-1]))
            index += 1

        csvfile.close()
        print "cost:",time.time()-cputime,"s"
        return AA,t



    def readtestfile(self, filename):
        cputime = time.time()
        print "start reading test file:",filename,"    ",
        csvfile = file(os.path.join(os.getcwd(), filename),"rb")
        reader = csv.reader(csvfile)
        data = []
        featureN = 0
        fname = []
        for line in reader:
            if line[0] == "Id":
                fname.extend( line )
                continue
            if featureN == 0:
                featureN = len(line)
        
            d = []
            for i in range(featureN):
                if fname[i] in self.discreteSet:
                    if line[i] == "":
                        d.append(-1)
                        continue
                    d.append(int(line[i]))
                    continue
                if fname[i] in self.continuesSet:
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
        
        dataN = len(data)
        AA = numpy.zeros((dataN,featureN-1))
        ID = []
        index = 0
        for d in data:
            dd = numpy.zeros(featureN-1)
            for i in range(featureN-1):
                dd[i] = (float(d[i+1]) - self.minv[i])/self.maxv[i]
                #dd.append(float(d[i+1]))
            AA[index,:] = dd
            ID.append(int(d[0]))
            index += 1
        
        csvfile.close()
        print "cost:",time.time() - cputime,"s"
        return ID,AA



# main entry
"""
pr = process()
x,y = pr.readtrainfile("train.csv")
dataN = x.shape[0]
featureN = x.shape[1]
print "data",dataN,"feature",featureN

""" 

