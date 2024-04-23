import numpy as np

def split(input,frac=0.666):
    input = input[np.argsort(input[:,-1])]
    no_plus = np.sum(input[:,-1]==1)
    no_minus = np.sum(input[:,-1]==0)
    minus = input[:no_minus]
    plus = input[no_minus:]
    np.random.shuffle(minus)
    np.random.shuffle(plus)
    minus_training=minus[:(int)(no_minus*frac)]
    minus_test=minus[(int)(no_minus*frac):]
    plus_training = plus[:(int)(no_plus*frac)]
    plus_test=plus[(int)(no_plus*frac):]
    training=np.concatenate((minus_training,plus_training),axis=0)
    test=np.concatenate((minus_test,plus_test),axis=0)
    np.random.shuffle(training)
    np.random.shuffle(test)
    return (training,test)

