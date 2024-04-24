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
nb_f1 = []
lr_f1 = []
for y in range(0,20):
    training,test = split(input)
    testX=test[:,:-1]
    testY=test[:,-1]
    nb_localacc=[]
    lr_localacc=[] 
    for x in frac:
        trainingFrac,_=split(training,x)
        trainingX=trainingFrac[:,:-1]
        trainingY=trainingFrac[:,-1]
        bayes=Naive_bayes(10,9,2)
        bayes.train(trainingX,trainingY)
        nb_localacc.append(bayes.F_beta(testX,testY))

        log_reg=Log_Reg(9)
        log_reg.train(trainingX,trainingY)
        lr_localacc.append(log_reg.F_beta(testX,testY))
    nb_f1.append(nb_localacc)
    lr_f1.append(lr_localacc)

nb_accuracy=np.mean(nb_f1,axis=0)
lr_accuracy=np.mean(lr_f1,axis=0)

plt.plot(frac, nb_accuracy, label='Naive Bayes')
plt.plot(frac, lr_accuracy, label='Logistic Regression')
plt.xlabel('Fraction of Training Data')
plt.ylabel('F1')
plt.title('F1 vs Fraction of Training Data')
plt.legend()
plt.show()

