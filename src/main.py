import numpy as np
import os
from naive_bayes import *


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "rp.data")
training_path= os.path.join(project_dir, "data", "training.data")
test_path = os.path.join(project_dir, "data", "test.data")

training = np.loadtxt(training_path)
test = np.loadtxt(test_path)
model,y_prob = naive_bayes(training)


success=0
fail=0
for i in test:
    if predict(model,y_prob,i[:-1])==i[-1]:
        success+=1
    else: 
        fail+=1
print(success/(success+fail))

#print(predict(model,y_prob,np.array([0,0,0,0,1,0,1,0,0])))