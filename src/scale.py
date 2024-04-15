import numpy as np

#all elements from X are from 1 to 10, we scale to 0,9
#we chenge Y from {4,2} to {1,0}
def scale(input):
    input-=1
    input[:,-1]-=1
    input[:,-1]/=2
    return input