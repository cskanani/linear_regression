import numpy as np

class LinearRegression:
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
        if(self.add_bais):
            X = (X - self._min) / (self._max - self._min)
            return np.dot(np.c_[np.ones(X.shape[0]), X] , self.th)
        else:
            X = (X - self._min) / (self._max - self._min)
            return X.dot(self.th)





dt = np.loadtxt('data/Admission_Predict.csv',delimiter=',',skiprows=1,usecols=range(1,9))
x_dt = dt[:,:7]
y_dt = dt[:,7:]

lr = LinearRegression(max_itr=10000,toll=0.0000001,learning_rate=1)
lr.fit(x_dt,y_dt)
th = lr.th

print('Original Value, Predicted Values')
print(np.c_[y_dt,lr.predict(x_dt)])
