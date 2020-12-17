# Trains Pong using ES with distributed computing 

import es 
import nn
import distrib 

import numpy as np 
import atari_py 
import gym 

import multiprocessing as mp 
import sys 
import time

env = gym.make('Pong-ram-v4') 

# define network architecture 
x = i = nn.Input((128,)) 
x = nn.Dense(6)(x) 
net = nn.Model(i, x) 
del x, i 

# vectorized weights and original shape information 
outw, outs = nn.get_vectorized_weights(net) 

# run Pong 
def fitness_pong(w, render: bool=False, steps=1000): 
    score = 0

    nn.set_vectorized_weights(net, w, outs) 

    for _ in range(3): 
        # env._max_episode_steps = steps
        obs = env.reset() 

        # fitness 
        s = 0

        while True: 
            close = False

            if render: 
                close = not env.render()
                # print(obs) 

            obs = obs / 256 

            # determine action 
            res = net.predict(np.expand_dims(obs, 0))[0]
            action = np.argmax(res) 

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
    # init ES 
    e = es.EvolutionStrategy(
        outw, 
        1.0, 
        1000, 
        15, 
        min_sigma=1e-3, 
        big_sigma=5e-2, 
        wait_iter=100000
    )

    # distributed training 
    pool = distrib.DistributedServer() 
    pool.start() 

    print("Waiting for connections...") 
    time.sleep(5) 
    print("Done!") 

    LENGTH = 1000
    times = 0 
    best = -float('inf') 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            # eval population 
            for ind in pop: 
                scores.append(pool.execute('pong_es', { 'w': ind })) 

            thread_scores = scores 
            scores = []

            ii = 0 
            for s in thread_scores: 
                scores.append(s.result())
                ii += 1 
                print("{} / {}".format(ii, len(thread_scores)), end='\r')

            e.tell(scores) 

            max_score = np.max(scores)  
            # if max_score > best: 
            if True: 
                if max_score > best: 
                    best = max_score 

                ind = pop[np.argmax(scores)] 
                # print(ind) 
                fitness_pong(ind, render=True) 

    except Exception as e: 
        print("Error while training:", e) 

    finally: 
        pool.stop() 