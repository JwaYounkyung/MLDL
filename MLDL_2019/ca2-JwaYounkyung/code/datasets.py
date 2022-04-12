from numpy import *
import util
import csv 
from sklearn.preprocessing import StandardScaler

class PriceDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """    
    tr_x = []   # X (data) of training set.
    tr_y = []   # Y (label) of training set.
    val_x = []  # X (data) of validation set.
    val_y = []  # Y (label) of validation set.
    ts_x = []
    ts_y = []

    def __init__(self):
        ## read the csv for training (price_data_tr.csv), 
        #                   val (price_data_val.csv)
        #                   and testing set (price_data_ts.csv)
        #
        ## CAUTION: the first row is the header 
        ##          (there is an option to skip the header 
        ##            when you read csv in python csv lib.)

        
        ### TODO: YOUR CODE HERE
        tr_x = []  ### TODO: YOUR CODE HERE
        tr_y = []  ### TODO: YOUR CODE HERE
        val_x = [] ### TODO: YOUR CODE HERE
        val_y = [] ### TODO: YOUR CODE HERE
        ts_x = []
        ts_y = []
        
        tr = open("/Users/jwa/Documents/ml2019fallca02-JwaYounkyung-master/dataset/price_data_tr.csv")
        tr_csv = csv.reader(tr)
        next(tr_csv)
        for i in tr_csv :
            tr_y.append(i[2])
            #del i[12]; del i[13]
            del i[5]
            del i[2]
            i[1] = i[1].replace("T000000","")
            tr_x.append(i)

        val = open("/Users/jwa/Documents/ml2019fallca02-JwaYounkyung-master/dataset/price_data_val.csv")
        val_csv = csv.reader(val)
        next(val_csv)
        for i in val_csv :
            val_y.append(i[2])
            #del i[12]; del i[13]
            del i[5]
            del i[2]
            i[1] = i[1].replace("T000000","")
            val_x.append(i)

        ts = open("/Users/jwa/Documents/ml2019fallca02-JwaYounkyung-master/dataset/price_data_ts.csv")
        ts_csv = csv.reader(ts)
        next(ts_csv)
        for i in ts_csv :
            ts_y.append(i[2])
            #del i[12]; del i[13]
            del i[5]
            del i[2]
            i[1] = i[1].replace("T000000","")
            ts_x.append(i)

        
        self.tr_x = tr_x
        self.val_x = val_x
        self.ts_x = ts_x

        self.tr_y = tr_y
        self.val_y = val_y
        
        '''
        scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x_scaled = scaler.transform(tr_x)
        scaler.fit(val_x)
        val_x_scaled = scaler.transform(val_x)
        scaler.fit(ts_x)
        ts_x_scaled = scaler.transform(ts_x)

        
        tr_y_m = array(tr_y, dtype=float).reshape(-1,1)
        val_y_m = array(val_y, dtype=float).reshape(-1,1)
        scaler.fit(tr_y_m)
        tr_y_scaled = scaler.transform(tr_y_m)
        scaler.fit(val_y_m)
        val_y_scaled = scaler.transform(val_y_m)
        

        self.tr_x = tr_x_scaled
        self.val_x = val_x_scaled
        self.ts_x = ts_x_scaled

        self.tr_y = tr_y_scaled
        self.val_y = val_y_scaled
        '''
        

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.val_x, self.val_y, self.ts_x, self.ts_y]
