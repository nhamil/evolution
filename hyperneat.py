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

        self.neat = neat.Neat(4, 2 * len(self.base_model.layers), args)

    # def get_coord(self, i_layer, index): 
    #     if i_layer == 0: 
    #         if len(index) == 1: 
    #             return (index[0] + 0.5) / self.input_shape[0] * 2 - 1, 0.5 
    #         elif len(index) == 2: 
    #             return (index[0] + 0.5) / self.input_shape[0] * 2 - 1, (index[1] + 0.5) / self.input_shape[1] * 2 - 1
    #         else: 
    #             raise Exception("Unsupport input dimension: {}".format(len(index))) 
    #     else: 
    #         i_layer -= 1 
    #         layer = self.base_model.layers[i_layer] 
    #         if isinstance(layer, nn.Dense): 
    #             return (index[0] + 0.5) / layer.output_shape[0] * 2 - 1, 0.5 

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

        for i in range(len(m.layers)): 
            layer = m.layers[i] 
            # print(i, layer.output_shape)
            if isinstance(layer, nn.Dense): 
                for j in range(layer.W.shape[1]): 
                    # print(j, layer.W.shape[1]) 
                    layer.b[j] = cppn.predict([
                        0.0, 
                        0.0, 
                        0.0, 
                        0.0 
                    ])[2*i + 0] * 20
                    for k in range(layer.W.shape[0]): 
                        layer.W[k,j] = cppn.predict([
                            (k + 0.5) / layer.W.shape[0] * 2 - 1, 
                            (j + 0.5) / layer.W.shape[1] * 2 - 1, 
                            0.0, 
                            0.0, 
                        ])[2*i + 1] * 20 
                # print(layer.get_weights()) 
            elif isinstance(layer, nn.Conv2D): 
                for i0 in range(layer.W.shape[0]): 
                    for i1 in range(layer.W.shape[1]): 
                        for i2 in range(layer.W.shape[2]): 
                            for i3 in range(layer.W.shape[3]): 
                                layer.W[i0,i1,i2,i3] = cppn.predict([
                                    (i0 + 0.5) / layer.W.shape[0] * 2 - 1, 
                                    (i1 + 0.5) / layer.W.shape[1] * 2 - 1, 
                                    (i2 + 0.5) / layer.W.shape[2] * 2 - 1, 
                                    (i3 + 0.5) / layer.W.shape[3] * 2 - 1, 
                                ])[2*i + 1] * 20 
            else: 
                # raise Exception("Unknown layer type: {}".format(type(layer)))
                pass 

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

    attempts = 100
    success = 0  
    gens = 0 

    for i in range(attempts): 
        hn = HyperNeat(m_cfg, {
            'n_pop': 500, 
            'max_species': 100, 
            'species_threshold': 1.0, 
            'clear_species': 15, 
            'prob_add_node': 0.0, 
            'prob_replace_weight': 0.1, 
            'prob_mutate_weight': 0.3, 
            'prob_toggle_conn': 0.2, 
            'prob_replace_activation': 0.2, 
            'std_new': 1.0, 
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