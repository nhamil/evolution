# Trains the lunar lander problem using ES 

import es 
import nn

import numpy as np 
import gym 

import multiprocessing as mp 
import sys 

env = gym.make('LunarLanderContinuous-v2') 

# define network architecture 
x = i = nn.Input((8,)) 
x = nn.Dense(2)(x) 
net = nn.Model(i, x) 
del x, i 

# vectorized weights and original shape information 
outw, outs = nn.get_vectorized_weights(net) 

# run lunar lander problem
def fitness_lander(w, render: bool=False, steps=1000): 
    score = 0

    nn.set_vectorized_weights(net, w, outs) 

    n = 10 
    if render: 
        n = 3

    for _ in range(n): 
        env._max_episode_steps = steps
        obs = env.reset() 

        # total reward (fitness score) 
        s = 0

        while True: 
            close = False

            if render: 
                close = not env.render()
                # print(obs) 

            # determine action to take 
            res = net.predict(np.expand_dims(obs, 0))[0]
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

    env.close() 

    return score / n

if __name__ == "__main__": 
    # init ES 
    e = es.EvolutionStrategy(
        outw, 
        5.0, 
        300, 
        10, 
        min_sigma=1e-3, 
        big_sigma=5e-2, 
        wait_iter=5
    )

    # multiprocessing 
    pool = mp.Pool() 

    LENGTH = 1000
    times = 0 
    best = 0 #-float('inf') 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            # eval population 
            for ind in pop: 
                scores.append(pool.apply_async(fitness_lander, ((ind, False, LENGTH)))) 

            thread_scores = scores 
            scores = []

            ii = 0 
            for s in thread_scores: 
                scores.append(s.get())
                ii += 1 
                print("{} / {}".format(ii, len(thread_scores)), end='\r')

            # scores = [s.get() for s in scores] 

            e.tell(scores) 

            max_score = np.max(scores)  
            if True: 
                if max_score > best: 
                    best = max_score 

                # show best individual 
                ind = pop[np.argmax(scores)] 
                fitness_lander(ind, render=True) 

    except Exception as e: 
        print("Error while training:", e) 