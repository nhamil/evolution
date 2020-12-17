# Trains the car racing problem using ES 

import es 
import nn
import comm 

import matplotlib.pyplot as plt 
import numpy as np 
import gym 

import multiprocessing as mp 
import sys 

env = gym.make('CarRacing-v0') 

# plt.ion() 

# define network architecture 
x = i = nn.Input((2*96//8*96//8*3//3,)) 
x = nn.Dense(20)(x) 
x = nn.Dense(3)(x) 
net = nn.Model(i, x) 
del x, i 

# vectorized weights and original shape information 
outw, outs = nn.get_vectorized_weights(net) 

# run car racing problem 
def fitness_car_race(w, render: bool=False, steps=1000): 
    score = 0

    nn.set_vectorized_weights(net, w, outs) 

    n = 2

    for _ in range(n): 
        # env._max_episode_steps = steps
        obs = env.reset() 
        last_obs = np.array(obs) / 255.0

        # net.clear() 

        # fitness 
        s = 0

        while True: 
            close = False

            if render: 
                close = not env.render()
                # print(obs) 

            obs = obs / 255.0 

            # if render: 
            #     plt.cla() 
            #     plt.imshow(obs[::8,::8,1]) 
            #     plt.pause(0.00001) 

            # determine action 
            res = net.predict(np.expand_dims(np.concatenate([
                last_obs[::8,::8,1].flatten(), 
                obs[::8,::8,1].flatten()
            ]), 0))[0]
            res = res * 2 - 1 
            action = res #np.argmax(res) 

            last_obs = obs 
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

    return score / n

if __name__ == "__main__": 
    # init ES 
    e = es.EvolutionStrategy(
        outw, 
        1.0, 
        100, 
        10, 
        min_sigma=1e-3, 
        big_sigma=1e1, 
        wait_iter=5
    )

    # multiprocessing 
    pool = mp.Pool() 

    LENGTH = 1000
    times = 0 
    best = -float('inf') 

    hist = open('car_es_hist.txt', 'w')  

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            # eval population 
            for ind in pop: 
                scores.append(pool.apply_async(fitness_car_race, ((ind, False, LENGTH)))) 

            thread_scores = scores 
            scores = []

            ii = 0 
            for s in thread_scores: 
                scores.append(s.get())
                ii += 1 
                print("{} / {}".format(ii, len(thread_scores)), end='\r')

            e.tell(scores) 

            max_score = np.max(scores)  
            if max_score > best: 
                best = max_score 

            # log score info 
            print("Writing...", end='') 
            hist.write("{}, {} \n".format(
                max_score, 
                np.mean(scores)
            ))
            hist.flush() 

            ind = pop[np.argmax(scores)] 
            
            # save best individual 
            f = open('models/car_{:03d}.es'.format(i+1), 'wb') 
            out = comm.encode(ind)
            f.write(out)
            f.close()  
            print("Done") 

            fitness_car_race(ind, render=True) 

    finally: 
        pass 