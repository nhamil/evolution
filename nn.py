import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras 
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer 

import numpy as np 

def shape_size(shape): 
    out = 1 
    for s in shape: 
        out *= s 
    return out 

def vectorize_weights(weights): 
    concat = [] 
    shapes = [] 
    for x in weights: 
        shapes.append(x.shape) 
        concat.append(x.flatten()) 
    out = np.concatenate(concat), shapes 
    return out 

def unvectorize_weights(data): 
    vec, shapes = data 
    weights = [] 
    start = 0 
    for s in shapes: 
        if s is not None and len(s) > 0: 
            sz = shape_size(s) 
            weights.append(vec[start:start+sz].reshape(*s))
            start += sz 
        else: 
            weights.append(None) 
    return weights 

def encode_model(model: Model, weights=True): 
    if weights: 
        return { 'config': model.get_config(), 'weights': model.get_weights() }
    else: 
        return { 'config': model.get_config() } 

def decode_model(data): 
    model = Sequential.from_config(data['config']['config'])

    if 'weights' in data: 
        model.set_weights(data['weights'])

    return model 
