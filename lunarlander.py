import neat 

import numpy as np 
import atari_py 
import gym 

import multiprocessing as mp 
import sys 

env = gym.make('LunarLanderContinuous-v2') 

# print('Input:', env.reset().shape) 
# print('Output:', env.action_space) 

# for x in gym.envs.registry.all(): 
#     print(x) 

# sys.exit(0) 

def fitness_lander(net: neat.Network, render: bool=False, steps=1000): 
    score = 0

    for _ in range(3): 
        env._max_episode_steps = steps
        obs = env.reset() 

        net.clear() 

        s = 0

        while True: 
            close = False

            if render: 
                close = not env.render()
                # print(obs) 

            res = net.predict(obs)
            res = res * 2 - 1 
            action = res #np.argmax(res) 

            obs, reward, done, _ = env.step(action)

            s += reward

            if done or close: 
                break

        score += s 

        if render: 
            print(s) 
            env.close() 

        if close: 
            break 

    return score / 3

if __name__ == "__main__": 
    neat_args = {
        'n_pop': 500, 
        'max_species': 30, 
        'species_threshold': 1.0, 
        'survive_threshold': 0.5, 
        'clear_species': 100, 
        'prob_add_node': 0.01, 
        'prob_add_conn': 0.05, 
        'prob_replace_weight': 0.01, 
        'prob_mutate_weight': 0.5, 
        'prob_toggle_conn': 0.01, 
        'prob_replace_activation': 0.1, 
        'std_new': 1.0, 
        'std_mutate': 0.01, 
        'activations': ['sigmoid'], 
        'dist_weight': 0.4, 
        'dist_activation': 1.0, 
        'dist_disjoint': 1.0  
    }

    n = neat.Neat(8, 2, neat_args) 

    pool = mp.Pool() 

    LENGTH = 1000
    times = 0 
    best = 0 #-float('inf') 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = n.ask() 

            for ind in pop: 
                # scores.append(fitness_lander(ind, render=False, steps=LENGTH)) 
                scores.append(pool.apply_async(fitness_lander, ((ind, False, LENGTH)))) 

            scores = [s.get() for s in scores] 

            n.tell(scores) 

            max_score = np.max(scores)  
            if max_score > best: 
                best = max_score 

                ind = pop[np.argmax(scores)] 
                # print(ind) 
                fitness_lander(ind, render=True) 

            if max_score == LENGTH: 
                times += 1 
            else: 
                times = 0 

            if times == 5: 
                print(ind) 
                break 

    except Exception as e: 
        print("Error while training:", e) 