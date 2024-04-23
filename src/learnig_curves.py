import numpy as np
import os
from models import *
from split import  *
from scale import *
import matplotlib.pyplot as plt



project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "rp.data")

input = np.loadtxt(data_path)
input=scale(input)

frac = [0.01,0.02,0.03,0.125,0.625,1]
nb_accuracy = []
lr_accuracy = []

for y in range(0,10):
    training,test = split(input)
    testX=test[:,:-1]
    testY=test[:,-1]
    nb_local=[]
    lr_local=[]    
    for x in frac:
        trainingFrac,_=split(training,x)
        trainingX=trainingFrac[:,:-1]
        trainingY=trainingFrac[:,-1]
        bayes=Naive_bayes(10,9,2)
        bayes.train(trainingX,trainingY)
        nb_local.append(bayes.evaluate(testX,testY))

        log_reg=Log_Reg(9)
        log_reg.train(trainingX,trainingY)
        lr_local.append(log_reg.evaluate(testX,testY))
    nb_accuracy.append(nb_local)
    lr_accuracy.append(lr_local)

nb_accuracy=np.mean(nb_accuracy,axis=0)
lr_accuracy=np.mean(lr_accuracy,axis=0)

plt.plot(frac, nb_accuracy, label='Naive Bayes')
plt.plot(frac, lr_accuracy, label='Logistic Regression')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Fraction of Training Data')
plt.legend()
plt.show()
