import sys 
from copy import deepcopy 
from typing import Union

import numpy as np 
import numpy.lib.stride_tricks as st 
import scipy as sp 
import scipy.signal as signal 
import matplotlib.pyplot as plt 

activations = {
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)), 
    'linear': lambda x: x, 
    'relu': lambda x: np.maximum(0.0, x) 
}

def convolve2D(imgs: np.ndarray, k: np.ndarray, stride: int=1):
    """
    a: (num, channel, y, x) \\
    k: (filter, channel, y, x) 
    """ 
    print() 
    print(imgs.shape) 
    print(k.shape) 
    print(stride) 

    sa = imgs[0].shape 
    sk = k.shape 
    s = stride
    so = ((sa[1]-sk[2])//s+1, (sa[2]-sk[3])//s+1)
    c_list = np.empty((len(imgs), sk[0], *so))

    for num in range(len(imgs)): 
        a = imgs[num]
        sa = a.shape 
        sta = a.strides
        sk = k.shape 
        so = ((sa[1]-sk[2])//s+1, (sa[2]-sk[3])//s+1)
        # print(so) 
        stride = (sta[1]*s, sta[2]*s, sta[0], sta[1], sta[2])
        b = st.as_strided(a, (so[0], so[1], sa[0], sk[2], sk[3]), stride)
        # print('b', b.shape) 
        c = c_list[num]
        # print('c', c.shape) 
        for i in range(len(k)): 
            ci = np.tensordot(b, k[i], axes=3)
            # print(i, ci.shape) 
            c[i] = np.reshape(ci, (1, *so))

    return c_list

class Input: 

    def __init__(self, output_shape: tuple): 
        self.output_shape = output_shape 
        self.output = None 

class Layer: 

    def __init__(self): 
        self._Layer__created = False 
        self.output = None

    def __call__(self, x: Union['Layer', Input, np.ndarray]): 
        if hasattr(self, '_Layer__created') and self._Layer__created: 
            self.output = self.eval(x) 
            return self.output 
        else: 
            out = deepcopy(self) 
            out._Layer__created = True 
            out.output_shape = out.build(x.output_shape) 
            out.input = x 
            return out 

    def get_config(self): 
        raise NotImplementedError 

    def get_weights(self): 
        raise NotImplementedError 

    def set_weights(self): 
        raise NotImplementedError 

    def build(self, in_shape: tuple): 
        raise NotImplementedError 

    def eval(self, x): 
        raise NotImplementedError 

class Dense(Layer): 

    def __init__(self, nodes: int, activation: str='sigmoid'): 
        super(Dense, self).__init__() 
        self.nodes = nodes 
        self.activation = activation

    def build(self, in_shape: tuple): 
        if len(in_shape) != 1: 
            raise Exception('Expected input shape to be 1 dimension') 

        self.W = np.full((in_shape[0], self.nodes), 0, dtype=np.float) 
        self.b = np.full(self.nodes, 0.0, dtype=np.float) 
        self.a = activations[self.activation] 
        return (self.nodes,)

    def eval(self, x): 
        print("Eval: {}".format(x))
        return self.a(np.dot(x, self.W) + self.b) 
        
    def get_weights(self): 
        return [np.copy(self.W), np.copy(self.b)] 

    def set_weights(self, weights): 
        self.W[:] = weights[0] 
        self.b[:] = weights[1] 

    def get_config(self): 
        return {
            'nodes': self.nodes, 
            'activation': self.activation 
        }

class Conv2D(Layer): 

    def __init__(self, filters: int, size: int, padding: str='same', stride: int=1, activation: str='linear'): 
        super(Conv2D, self).__init__() 
        self.filters = filters 
        self.size = size 
        self.padding = padding 
        self.stride = stride 
        self.activation = activation 

        if padding == 'same' and stride > 1: 
            raise Exception("Same padding and stride > 1 is not currently supported")

    def build(self, in_shape: tuple): 
        if len(in_shape) != 3: 
            raise Exception('Expected input shape to be 2 dimensions') 

        self.W = np.full((self.filters, in_shape[0], self.size, self.size), 0, dtype=np.float) 
        self.a = activations[self.activation] 

        out_shape = None 
        if self.padding == 'same': 
            out_shape = (
                self.filters, 
                in_shape[1], 
                in_shape[2] 
            )
        else: 
            out_shape = (
                self.filters, 
                (in_shape[1] - self.size) // self.stride + 1, 
                (in_shape[2] - self.size) // self.stride + 1, 
            )
        self.out_shape = out_shape 
        print(self.out_shape) 

        return out_shape 

    def eval(self, x): 
        # return signal.convolve2d(x[0,0,:,:], self.W[0,0,:,:])
        # st.as_strided()
        if self.padding == 'same': 
            pass 
        else: 
            return self.a(convolve2D(x, self.W, self.stride)) 

    def get_weights(self): 
        return [np.copy(self.W)] 

    def set_weights(self, weights): 
        self.W[:] = weights[0] 

    def get_config(self): 
        return {
            'filters': self.filters, 
            'size': self.size, 
            'padding': self.padding, 
            'stride': self.stride, 
            'activation': self.activation  
        }

class Flatten(Layer): 

    def __init__(self): 
        super(Flatten, self).__init__() 

    def build(self, in_shape: tuple): 
        out_n = 1 
        for x in in_shape: 
            out_n *= x 
        self.out_n = out_n 
        return (out_n,) 

    def eval(self, x): 
        return np.reshape(x, (-1, self.out_n))

    def get_weights(self): 
        return [] 

    def set_weights(self, weights): 
        pass 

    def get_config(self): 
        return {} 

def get_all_layers(layer: Layer, out: list=None): 
    if out is None: 
        out = [] 
    out.append(layer) 
    if hasattr(layer, 'input') and not isinstance(layer.input, Input): 
        get_all_layers(layer.input, out) 
    return out[::-1] 

class Model: 

    layer_types = {
        'Dense': Dense, 
        'Flatten': Flatten, 
        'Conv2D': Conv2D 
    }

    def __init__(self, input: Input, output: Layer): 
        self.__input = input 
        self.__output = output 
        self.input_shape = self.__input.output_shape 
        self.output_shape = self.__output.output_shape 
        self.layers = get_all_layers(output) 

    @staticmethod
    def from_config(cfg: dict): 
        i = Input(cfg['input_shape']) 
        cur = i
        for layer, c in cfg['layers']: 
            cur = Model.layer_types[layer](**c)(cur) 
        return Model(i, cur) 

    def predict(self, x: np.ndarray): 
        if len(x.shape) != len(self.__input.output_shape) + 1: 
            raise Exception("Expected input shape of {} but got {}".format((None, *self.__input.output_shape), x.shape))
        self.__input.output = x 
        for layer in self.layers: 
            x = layer(layer.input.output) 
        return x 

    def get_config(self): 
        return {
            'input_shape': self.__input.output_shape, 
            'output_shape': self.__output.output_shape, 
            'layers': [
                (type(layer).__name__, layer.get_config()) for layer in self.layers 
            ]
        }

    def get_weights(self): 
        return [
            layer.get_weights() for layer in self.layers 
        ]

    def set_weights(self, weights): 
        for i in range(len(self.layers)): 
            self.layers[i].set_weights(weights[i]) 
