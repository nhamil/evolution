import es 
import nn 

import numpy as np 

import multiprocessing as mp 

x = i = nn.Input((2,)) 
x = nn.Dense(2)(x) 
x = nn.Dense(1)(x) 
net = nn.Model(i, x) 
del x, i 

outw, outs = nn.get_vectorized_weights(net) 

def fitness_xor(w: np.ndarray): 
    total, p = 0, 2

    nn.set_vectorized_weights(net, w, outs)
    out = net.predict(np.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1] 
    ]))

    total += np.power(0 - out[0, 0], p) 
    total += np.power(1 - out[1, 0], p) 
    total += np.power(1 - out[2, 0], p) 
    total += np.power(0 - out[3, 0], p) 
    return 4 - total 

if __name__ == "__main__": 
    e = es.EvolutionStrategy(
        outw, 
        1.0, 
        500, 
        10, 
        min_sigma=1e-3, 
        big_sigma=1e1, 
        wait_iter=100
    )

    # pool = mp.Pool() 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            for ind in pop: 
                scores.append(fitness_xor(ind)) 
                # scores.append(pool.apply_async(fitness_xor, ((ind,)))) 

            # scores = [s.get() for s in scores] 

            e.tell(scores) 

            max_score = np.max(scores)  
            
            # if max_score >= 3.9: 
            if i == 99: 
                ind = pop[np.argmax(scores)] 
                print(ind) 
                nn.set_vectorized_weights(net, ind, outs)
                out = net.predict(np.array([
                    [0, 0], 
                    [0, 1], 
                    [1, 0], 
                    [1, 1] 
                ]))
                print(out) 
                break 

    finally: 
        pass 
    # except Exception as e: 
    #     print("Error while training:", e) 