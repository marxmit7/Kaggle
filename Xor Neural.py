import numpy as np
import time

# variables
h_hidden=10
h_in=10
h_out=10

#sample data
n_sample=300

#hpyerparameters
learning_rate= 0.001
momentum=0.9
#non deterministic seeding
np,.random.seed(0)

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def tanh_prime(x):
    return 1-np.tanh(x)**2

#input data, transpose, layer 1, layer 2,bias 1, bias2
def train(x,t,V,W,bv,bw):
    
