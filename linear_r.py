#TODO: add tolerance feature

import numpy as np

#class for linear regression
class LinearRegression:
    '''
        INPUT:
        normalize(bool): apply min-max normalization or not
        add_bais(bool): add bais term or not
        learning_rate(float): learning rate for algorithm
        toll(float): how much tolerance algorithm can accept, can use for early stopping
        max_itr(int): number of maximum iteration after which algorithm will stop
    '''
    def __init__(self,normalize=True,add_bais=True,learning_rate=0.1,toll=0.0001,max_itr=100):
        self.normalize = normalize
        self.add_bais = add_bais
        self.learning_rate = learning_rate
        self.toll = toll
        self.max_itr = max_itr
        self.th = 0
        self._min = 0
        self._max = 0
        
    def fit(self,X,y):
        '''
        DESCRIPTION:
        takes input data and corrosponding values and returns the parametes for hypothesis
        INPUT:
        X(np array): input data
        y(np array): output values
        OUTPUT:
        returns th, a np array of shape (X.shape[1],1) with parameter values for hypothesis
        '''
        self._min = X.min()
        self._max = X.max()
        m = y.shape[0]
        
        if(self.normalize):
            X = (X - self._min) / (self._max - self._min)
        
        if(self.add_bais):
            X = np.c_[np.ones(m), X]
            
        th = np.zeros((X.shape[1],1))
        
        nth = th - (self.learning_rate / m) * (X.T).dot(X.dot(th) - y)
        
        i = 0
        while (i < self.max_itr):
            th = th - (self.learning_rate / m) * (X.T).dot(X.dot(th) - y)
            i+=1
        
        self.th = th
        return th

    
    def predict(self,X):
        '''
        DESCRIPTION:
        takes input data and predicts corrosponding values
        INPUT:
        X(np array): input data
        OUTPUT:
        returns a np array of predicted values for input data
        '''
        if(self.add_bais):
            X = (X - self._min) / (self._max - self._min)
            return np.dot(np.c_[np.ones(X.shape[0]), X] , self.th)
        else:
            X = (X - self._min) / (self._max - self._min)
            return X.dot(self.th)




#loading and splitting
dt = np.loadtxt('data/Admission_Predict.csv',delimiter=',',skiprows=1,usecols=range(1,9))
np.random.shuffle(dt)
n = dt.shape[0]
x_train = dt[:int(n*.8),:7]
y_train = dt[:int(n*.8),7:]
x_test = dt[int(n*.8):,:7]
y_test = dt[int(n*.8):,7:]

#creating model object for training
lr = LinearRegression(max_itr=10000,toll=0.0000001,learning_rate=1)
lr.fit(x_train,y_train)
th = lr.th

#prediciting on test data using learned model
pv = lr.predict(x_test)
print('Original Value, Predicted Values on test data')
print(np.c_[y_test,pv])

print('Mean squared error is : {}'.format(((pv-y_test)**2).mean()))
