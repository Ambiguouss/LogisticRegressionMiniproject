import numpy as np
import os
from models import *

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "rp.data")
training_path= os.path.join(project_dir, "data", "training.data")
test_path = os.path.join(project_dir, "data", "test.data")

training = np.loadtxt(training_path)
test = np.loadtxt(test_path)




trainingX=training[:,:-1]
trainingY=training[:,-1]
testX=test[:,:-1]
testY=test[:,-1]


bayes=Naive_bayes(10,9,2)
bayes.train(trainingX,trainingY)
log_reg=Log_Reg(9)
log_reg.train(trainingX,trainingY)


print(bayes.accuracy(testX,testY))
print(log_reg.accuracy(testX,testY))

print(bayes.F_beta(testX,testY))
print(log_reg.F_beta(testX,testY))
