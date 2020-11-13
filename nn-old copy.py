import numpy as np 

import comm 

DENSE = 1 

def shape_size(shape): 
    out = 1 
    for s in shape: 
        out *= s 
    return out 

def weights_to_vector(weights): 
    concat = [] 
    shapes = [] 
    for lst in weights: 
        shapes.append([w.shape for w in lst]) 
        concat.extend([w.flatten() for w in lst]) 
    out = np.concatenate(concat), shapes 
    return out 

def vector_to_weights(data): 
    vec, shapes = data 
    weights = [] 
    start = 0 
    for lst in shapes: 
        out = [] 
        for s in lst: 
            sz = shape_size(s) 
            out.append(vec[start:start+sz].reshape(*s))
            start += sz 
        weights.append(out) 
    return weights 

def dict_to_network(data={}): 
    in_shape = data['in']
    layers = data['layers'] 
    weights = data['weights'] 
    mb = ModelBuilder(in_shape) 
    for layer in layers: 
        getattr(mb, layer[0])(*layer[1:])
    net = mb.build() 
    net.set_weights(weights) 
    return net

def network_to_dict(net, weights=None):
    if weights is None: 
        weights = net.get_weights() 
    out = {} 
    out['in'] = net.in_shape 
    out['weights'] = weights
    layers = [] 
    for layer in net.layers: 
        t = type(layer) 
        l = None 
        if t == Dense: 
            l = ['dense', layer.out_shape, layer.activation] 
        else: 
            raise Exception("Unknown layer type: {}".format(t))
        layers.append(l) 
    out['layers'] = layers 
    return out 

class ModelBuilder: 

    def __init__(self, in_shape): 
        self.in_shape = in_shape 
        self.out_shape = in_shape 
        self.layers = [] 
        self.size = 0

    def _add(self, layer, out): 
        self.layers.append(layer) 
        self.size += layer[1] 
        self.out_shape = out 
        return self 

    def dense(self, neurons, activation='sigmoid'): 
        return self._add((DENSE, self.out_shape * neurons + neurons, self.out_shape, neurons, activation), neurons)

    def build(self, weights=None): 
        return Model(self.in_shape, self.layers, weights) 

class Layer: 

    def eval(self, x): 
        raise NotImplementedError() 

    def get_weights(self): 
        raise NotImplementedError() 

    def set_weights(self, x): 
        raise NotImplementedError() 

class Dense(Layer): 

    def __init__(self, in_shape, out_shape, activation, weights=None): 
        self.in_shape = in_shape 
        self.out_shape = out_shape 
        self.activation = activation 

        self.W = np.full((in_shape, out_shape), 0) 
        self.b = np.full(out_shape, 0)  

        if activation == 'sigmoid': 
            self.a = lambda x: 1.0 / (1.0 + np.exp(-x)) 
        else: 
            raise Exception("Unknown activation function") 

    def eval(self, x): 
        return self.a(np.dot(x, self.W) + self.b)

    def get_weights(self): 
        return [self.W, self.b] 

    def set_weights(self, x): 
        self.W[:] = x[0] 
        self.b[:] = x[1] 

class Model: 

    def __init__(self, in_shape, layers, weights=None): 
        self.in_shape = in_shape 
        self.out_shape = in_shape 
        self.layers = [] 
        for layer in layers: 
            out = None 
            t = layer[0] 
            if t == DENSE: 
                out = Dense(*layer[2:]) 
            else: 
                raise Exception("Unknown layer type") 
            self.layers.append(out) 
            self.out_shape = out.out_shape 

    def predict(self, x): 
        out = x 
        for layer in self.layers: 
            out = layer.eval(out) 
        return out 

    def get_layer(self, index): 
        return self.layers[index] 

    def get_weights(self): 
        return weights_to_vector([l.get_weights() for l in self.layers]) 

    def set_weights(self, w): 
        w = vector_to_weights(w) 
        for i in range(len(self.layers)): 
            self.layers[i].set_weights(w[i])

if __name__ == "__main__": 
    x = ModelBuilder(2) 
    x.dense(2) 
    x.dense(1) 
    model = x.build() 

    model.get_layer(0).set_weights([
        np.array([
            [20, -20],  
            [20, -20]
        ]), 
        np.array([-10, 30])
    ])

    model.get_layer(1).set_weights([
        np.array([
            [20], 
            [20]
        ]), 
        np.array([-30])
    ])

    out = model.predict(np.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1]
    ]))

    print(out.shape) 
    for i in range(len(out)): 
        print("{:0.3f}".format(out[i, 0]))

    d = network_to_dict(model)
    print(d) 
    n = dict_to_network(d) 
    print(n.get_weights()) 