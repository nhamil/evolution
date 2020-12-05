import es 
import nn
import comm 

import matplotlib.pyplot as plt 
import numpy as np 
import atari_py 
import gym 

import multiprocessing as mp 
import sys 

env = gym.make('CarRacing-v0') 

# print('Input:', env.reset().shape) 
# print('Output:', env.action_space) 

# for x in gym.envs.registry.all(): 
#     print(x) 

# sys.exit(0) 

plt.ion() 

x = i = nn.Input((2*96//8*96//8*3//3,)) 
x = nn.Dense(20)(x) 
x = nn.Dense(3)(x) 
net = nn.Model(i, x) 
del x, i 

outw, outs = nn.get_vectorized_weights(net) 

def fitness_car_race(w, render: bool=False, steps=1000): 
    score = 0

    nn.set_vectorized_weights(net, w, outs) 

    n = 2

    for _ in range(n): 
        # env._max_episode_steps = steps
        obs = env.reset() 
        last_obs = np.array(obs) / 255.0

        # net.clear() 

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
    e = es.EvolutionStrategy(
        outw, 
        1.0, 
        100, 
        10, 
        min_sigma=1e-3, 
        big_sigma=1e1, 
        wait_iter=5
    )

    pool = mp.Pool() 

    LENGTH = 1000
    times = 0 
    best = -float('inf') 

    print("Reading") 
    f = open('models/car_{:03d}.es'.format(67), 'rb') 
    in_data = comm.decode(f.read())
    print(in_data) 

    # print(ind) 
    fitness_car_race(in_data, render=True) 

    print("Done") 
    sys.exit(0) 

    hist = open('models/car_es_hist.txt', 'w')  

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            for ind in pop: 
                # scores.append(fitness_car_race(ind, render=False, steps=LENGTH)) 
                scores.append(pool.apply_async(fitness_car_race, ((ind, False, LENGTH)))) 

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
            if max_score > best: 
                best = max_score 

            print("Writing...", end='') 
            hist.write("{}, {}, \n".format(
                max_score, 
                np.mean(scores)
            ))
            hist.flush() 

            ind = pop[np.argmax(scores)] 
            
            f = open('models/car_{:03d}.es'.format(i+1), 'wb') 
            out = comm.encode(ind)
            f.write(out)
            f.close()  
            print("Done") 

            # print("Reading") 
            # f = open('models/car_{:03d}.neat'.format(i+1), 'rb') 
            # in_data = f.read() 
            # net = n.create_network(neat.Genome.load(comm.decode(in_data))) 

            # print(ind) 
            fitness_car_race(ind, render=True) 

            if max_score >= LENGTH: 
                times += 1 
            else: 
                times = 0 

            if times == 5: 
                print(ind) 
                break 

    # except Exception as e: 
    #     print("Error while training:", e) 
    finally: 
        pass 