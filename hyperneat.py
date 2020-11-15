from copy import deepcopy 

import numpy as np 

import neat 
import nn 
import util 

class HyperNeat: 

    args = {
        'n_pop', 
        'n_elite', 
        'clear_species', 
        'species_threshold', 
        'survive_threshold', 
        'max_species', 
        'dist_disjoint', 
        'dist_weight', 
        'dist_activation', 
        'std_mutate', 
        'std_new', 
        'prob_mutate_weight', 
        'prob_replace_weight', 
        'prob_add_conn', 
        'prob_add_node', 
        'prob_toggle_conn', 
        'prob_replace_activation', 
        'custom_activations', 
        'activations'
    }

    def __init__(self, model_cfg: dict, args: dict={}): 
        for key in args: 
            if key not in HyperNeat.args: 
                raise Exception("Unknown argument: {}".format(key)) 

        self.model_cfg = deepcopy(model_cfg) 
        
        tmp = nn.Model.from_config(model_cfg) 
        self.base_model = tmp 

        print("Creating HyperNEAT with {} inputs and {} outputs".format(
            util.shape_size(tmp.input_shape), 
            util.shape_size(tmp.output_shape)
        )) 

        self.input_shape = util.shape_size(tmp.input_shape) 
        self.output_shape = util.shape_size(tmp.output_shape)

        self.neat = neat.Neat(4, 1, args)

    def create_network(self, cppn: neat.Network): 
        # print(cppn) 

        m = None 
        try: 
            m = nn.Model.from_config(self.model_cfg) 
        except Exception as e: 
            print("Error time: {}".format(e))
            print(m) 
            print(nn) 
            print(nn.Model)
            print(self.model_cfg) 
            import sys 
            sys.exit() 
        layer_mul = 1.0 / (len(m.layers)) 

        for i in range(len(m.layers)): 
            w = i * layer_mul 
            layer = m.layers[i] 
            if isinstance(layer, nn.Dense): 
                for j in range(layer.W.shape[1]): 
                    layer.b[j] = cppn.predict([
                        0.5, 
                        j / layer.W.shape[1], 
                        1.0, 
                        w 
                    ]) * 20
                    for i in range(layer.W.shape[0]): 
                        layer.W[i,j] = cppn.predict([
                            i / layer.W.shape[0], 
                            j / layer.W.shape[1], 
                            0.0, 
                            w 
                        ]) * 20 
                # print(layer.get_weights()) 
            else: 
                raise Exception("Unknown layer type: {}".format(type(layer)))

        return m 

    def ask(self): 
        cppns = self.neat.ask() 
        self.gen = self.neat.gen 
        return [self.create_network(cppn) for cppn in cppns]

    def tell(self, scores: list): 
        self.neat.tell(scores)  
        self.gen = self.neat.gen 

if __name__ == "__main__":
    i = x = nn.Input((2,)) 
    x = nn.Dense(2, activation='sigmoid')(x) 
    x = nn.Dense(1, activation='sigmoid')(x) 
    x = nn.Model(i, x) 
    m_cfg = x.get_config() 
    del i, x 

    pop = None 
    fit = None

    attempts = 1
    success = 0  
    gens = 0 

    for i in range(attempts): 
        hn = HyperNeat(m_cfg, {
            'n_pop': 100, 
            'max_species': 100, 
            'species_threshold': 1.5, 
            'clear_species': 15, 
            'prob_add_node': 0.3, 
            'prob_replace_weight': 0.4, 
            'prob_mutate_weight': 0.9, 
            'prob_toggle_conn': 0.3, 
            'prob_replace_activation': 0.4, 
            'std_new': 10.0, 
            'std_mutate': 0.1 
        }) 

        for _ in range(100): 
            pop = hn.ask() 
            fit = [] 

            for net in pop: 
                f = 4 
                p = 2.0
                f -= np.power(np.abs(net.predict(np.array([[0, 0]])) - 0), p) 
                f -= np.power(np.abs(net.predict(np.array([[0, 1]])) - 1), p) 
                f -= np.power(np.abs(net.predict(np.array([[1, 0]])) - 1), p) 
                f -= np.power(np.abs(net.predict(np.array([[1, 1]])) - 0), p) 
                f = np.sum(f) 
                fit.append(f) 

            print('Iteration {}: '.format(i+1), end='') 
            hn.tell(fit) 

            if np.max(fit) >= 3.8: 
                print("Early stopping") 
                gens += hn.gen 
                success += 1
                break 

    i = np.argmax(fit) 
    score = fit[i] 
    net = pop[i] 
    print("Score: {:0.3f}, Net: {}".format(score, net))
    print("- [0 0] = {:0.3f}".format(net.predict(np.array([[0, 0]]))[0, 0]))
    print("- [0 1] = {:0.3f}".format(net.predict(np.array([[0, 1]]))[0, 0]))
    print("- [1 0] = {:0.3f}".format(net.predict(np.array([[1, 0]]))[0, 0]))
    print("- [1 1] = {:0.3f}".format(net.predict(np.array([[1, 1]]))[0, 0]))
    print(net) 

    if success > 0: print("Success rate: {:0.2f}%, Average generations: {:0.2f}".format(100 * success / attempts, gens / success))
    print(net.get_config()) 
    print(net.get_weights()) 