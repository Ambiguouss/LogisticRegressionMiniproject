import numpy as np

class Model:
    def evaluate(self,testX,testY):
        results = np.apply_along_axis(self.predict,axis=1,arr=testX)
        success = np.count_nonzero(results==testY)
        return success/results.shape[0]

class Naive_bayes(Model):
    def __init__(self,x_values,features,y_values):
        self.x_values=x_values
        self.features=features
        self.y_values=y_values
        self.result = np.zeros((x_values,features,y_values))
        self.y_prob=np.zeros(y_values)
    
    def train(self,trainingX,trainingY):
        for i in range(self.y_values):
            self.y_prob[i]=(1+np.sum(trainingY==i))/(2+trainingY.shape[0])
        for i in range(self.x_values):
            for j in range(self.features):
                for k in range(self.y_values):
                    self.result[i][j][k]=(
                        (1+np.sum((trainingX[:,j]==i)
                           &(trainingY==k)))/(2+self.y_prob[k]))
    
    def predict(self,test):
        probability = self.y_prob[1]
        for i in range(test.size):
            probability*=self.result[(int)(test[i])][i][1]
        denominator1=self.y_prob[1]
        denominator2=self.y_prob[0]
        for i in range(test.size):
            denominator1*=self.result[(int)(test[i])][i][1]
            denominator2*=self.result[(int)(test[i])][i][0]
        return (int)(probability/(denominator1+denominator2)>0.5)

class Log_Reg(Model):
    def __init__(self,features):
        self.features=features
        self.theta=np.zeros((features))

    def train(self,trainingX,trainingY,step=0.1,iterations=100000):
        for _ in range(iterations):
            a=(trainingY-(1.0/(1.0+np.exp(trainingX@(-self.theta)))))
            gradient = (np.transpose(trainingX)@a)
            self.theta += step*gradient
    
    def predict(self,test):
        return (int)(1.0/(1+np.exp(test@(-self.theta))))>0.5