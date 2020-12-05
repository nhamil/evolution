import neat 

import numpy as np 

import multiprocessing as mp 

def fitness_xor(net: neat.Network): 
    total, p = 0, 2
    total += np.power(0 - net.predict([0, 0]), p) 
    total += np.power(1 - net.predict([0, 1]), p) 
    total += np.power(1 - net.predict([1, 0]), p) 
    total += np.power(0 - net.predict([1, 1]), p) 
    total = total[0]
    return 4 - total 

if __name__ == "__main__": 
    neat_args = {
        'n_pop': 500, 
        'max_species': 30, 
        'species_threshold': 1.0, 
        'survive_threshold': 0.5, 
        'clear_species': 15, 
        'prob_add_node': 0.001, 
        'prob_add_conn': 0.01, 
        'prob_replace_weight': 0.01, 
        'prob_mutate_weight': 0.5, 
        'prob_toggle_conn': 0.01, 
        'prob_replace_activation': 0.1, 
        'std_new': 1.0, 
        'std_mutate': 1.0, 
        'activations': ['sigmoid'], 
        'dist_weight': 0.05, 
        'dist_activation': 1.0, 
        'dist_disjoint': 1.0  
    }

    n = neat.Neat(2, 1, neat_args) 

    pool = mp.Pool() 

    try: 
        for i in range(100): 
            scores = [] 
            pop = n.ask() 

            for ind in pop: 
                # scores.append(fitness_xor(ind)) 
                scores.append(pool.apply_async(fitness_xor, ((ind,)))) 

            scores = [s.get() for s in scores] 

            n.tell(scores) 

            max_score = np.max(scores)  
            
            # if max_score >= 3.99999: 
            if i == 99: 
                ind = pop[np.argmax(scores)] 
                print(ind) 
                print(ind.predict([0, 0])) 
                print(ind.predict([0, 1])) 
                print(ind.predict([1, 0])) 
                print(ind.predict([1, 1])) 
                break 

    except Exception as e: 
        print("Error while training:", e) 