
import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 

import lasagne

import theano
import theano.tensor as T

#import lasagne
# change the order of lasagne importing

import cPickle as pickle
import gzip

#import binary_ops
import binary_net

from pylearn2.datasets.cifar10s import CIFAR10 

from collections import OrderedDict

def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)



if __name__ == "__main__":
    
    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    
    # BinaryConnect    
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    print('Loading CIFAR-10 dataset...')
    
    test_set = CIFAR10(which_set="test", start=0,stop = 5000)
    #test_set = CIFAR10(which_set="test")
        
    # bc01 format
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # flatten targets
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])
    # for hinge loss
    test_set.y = 2* test_set.y - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    
    layer_beforepad = lasagne.layers.get_output(cnn, deterministic=True)        
    
    # 128C3-128C3-P2  
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding       

    layer_afterpad = lasagne.layers.get_output(cnn, deterministic=True)       
    
    
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    layer_afterconv = lasagne.layers.get_output(cnn, deterministic=True)   
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding   
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 256C3-256C3-P2  
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding   
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding   
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 512C3-512C3-P2  
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding               
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
                  
    cnn = lasagne.layers.PadLayer(cnn, width=1, val=1)
    # swd ~~~add padding layer before conv2d to add +1 padding   
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=512, 
            filter_size=(3, 3),
            pad=0,
            nonlinearity=lasagne.nonlinearities.identity,
	    b=None)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    # print(cnn.output_shape)
    
    # 1024FP-1024FP-10FP            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024,
	        b=None) 
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024,
	        b=None)
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10,
	        b=None) 
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a second function computing the validation accuracy:
    val_fn = theano.function([input, target], test_err)
    
    
    observe_fn0 = theano.function([input], layer_beforepad)   
    observe_fn1 = theano.function([input], layer_afterpad)  
    observe_fn2 = theano.function([input], layer_afterconv)  


    print("Loading the trained parameters and binarizing the weights...")
    
    # Load parameters
    with np.load('TrainModel.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

    # Binarize the weights
    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        #print param.name, param.get_value().shape
        if param.name == "W":
            param.set_value(SignNumpy(param.get_value()))
    
    print('Running...')
    
    start_time = time.time()
    
    test_error = val_fn(test_set.X,test_set.y)*100.
    print("test_error = " + str(test_error) + "%")
    
    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")
    
    
    
    
    print observe_fn0(test_set.X).shape
    
    print observe_fn1(test_set.X).shape
    
    print observe_fn2(test_set.X).shape
    #
    
    observe_target=observe_fn0(test_set.X)
    print observe_target[0,0,:,:]
    
    observe_target1=observe_fn1(test_set.X)
    print observe_target1[0,0,:,:]


