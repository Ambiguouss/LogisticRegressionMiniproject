import numpy as np


#result[i][j][k] = p(x_j==i|y==k)
def naive_bayes(training):
    x_values=10 #values of x_i are from 1 to 10
    features=9
    y_values=2 
    y_prob=np.zeros(y_values)
    for i in range(y_values):
        y_prob[i]=np.sum(training[:,-1]==i)/training.shape[0]
    result = np.zeros((x_values,features,y_values))
    for i in range(x_values):
        for j in range(features):
            for k in range(y_values):
                result[i][j][k]=(
                    np.sum((training[:,j]==i)
                           &(training[:,-1]==k))/y_prob[k])
    return result,y_prob


def predict(model,y_prob, input):
    probability = y_prob[1]
    for i in range(input.size):
        probability*=model[(int)(input[i])][i][1]
    denominator1=y_prob[1]
    denominator2=y_prob[0]
    for i in range(input.size):
        denominator1*=model[(int)(input[i])][i][1]
        denominator2*=model[(int)(input[i])][i][0]
    return (int)(probability>0.5*(denominator1+denominator2))