import numpy as np
import os
from scale import *
from split import *

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "rp.data")
training_path= os.path.join(project_dir, "data", "training.data")
test_path = os.path.join(project_dir, "data", "test.data")
input = np.loadtxt(data_path)
input=scale(input)

training,test = split(input)
np.savetxt(training_path,training)
np.savetxt(test_path,test)

