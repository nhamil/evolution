import neat 
import comm 

import matplotlib.pyplot as plt 
import numpy as np 

import multiprocessing as mp 
import sys 

from chess_eval import * 

env = ChessEnv() 

def fitness_chess(net: neat.Network, render: bool=False): 
    score = 0

    n = 1

    for _ in range(n): 
        # env._max_episode_steps = steps
        mv, obs = env.reset(render) 

        s = 0
        turn = 0

        while True: 
            turn += 1
            # print("Turn {}".format(turn), end='\r')
            close = False

            scores = [] 
            for i in range(len(mv)): 
                net.clear() 
                val = net.predict(obs[i].flatten())[0]
                # print(mv[i], val)
                scores.append(val) 

            action = mv[np.argmax(scores)]
            # print('action', action) 

            mv, obs, reward, done = env.step(action)

            s += reward

            if done or close: 
                break

        score += s + turn

        if render: 
            f.write("Score: {}, Turns: {}\n".format(s, turn)) 
        
        if close: 
            break 

    if np.abs(score / n) < 5: 
        print('zero score', score)

    return score / n

if __name__ == "__main__": 
    neat_args = {
        'n_pop': 500, 
        'max_species': 30, 
        'species_threshold': 4.0, 
        'survive_threshold': 0.5, 
        'clear_species': 100, 
        'prob_add_node': 0.05, 
        'prob_add_conn': 0.1, 
        'prob_replace_weight': 0.05, 
        'prob_mutate_weight': 0.5, 
        'prob_toggle_conn': 0.03, 
        'prob_replace_activation': 0.0, 
        'std_new': 3.0, 
        'std_mutate': 0.01, 
        'activations': ['sigmoid'], 
        'dist_weight': 0.5, 
        'dist_activation': 1.0, 
        'dist_disjoint': 1.0  
    }

    n = neat.Neat(8*8*6, 1, neat_args) 

    pool = mp.Pool() 

    LENGTH = 1000
    times = 0 
    best = -float('inf') 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = n.ask() 

            for ind in pop: 
                # scores.append(fitness_chess(ind, render=None)) 
                scores.append(pool.apply_async(fitness_chess, ((ind, None)))) 

            scores = [s.get() for s in scores] 

            n.tell(scores) 

            max_score = np.max(scores)  
            if max_score > best: 
                best = max_score 

            ind = pop[np.argmax(scores)] 
            f = open('chess_logs/chess_neat_{:03d}.txt'.format(i+1), 'w')
            fitness_chess(ind, render=f)
            f.close()  

    # except Exception as e: 
    #     print("Error while training:", e) 
    finally: 
        pass 