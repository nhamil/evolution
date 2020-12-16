import es 
import nn

import numpy as np 
import gym 

import multiprocessing as mp 
import sys 

env = gym.make('CartPole-v1') 

# print('Input:', env.reset().shape) 
# print('Output:', env.action_space) 
# sys.exit(0) 

# for x in gym.envs.registry.all(): 
#     print(x) 

x = i = nn.Input((4,)) 
x = nn.Dense(2)(x) 
net = nn.Model(i, x) 
del x, i 

outw, outs = nn.get_vectorized_weights(net) 

def fitness_cartpole(w: np.ndarray, render: bool=False, steps=1000): 
    score = 0

    nn.set_vectorized_weights(net, w, outs)
    
    n = 10 
    if render: 
        n = 1 

    for _ in range(n): 
        env._max_episode_steps = steps
        obs = env.reset() 

        s = 0

        while True: 
            close = False

            if render: 
                close = not env.render()
                # print(obs) 

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

    return score / n

if __name__ == "__main__": 
    e = es.EvolutionStrategy(
        outw, 
        0.1, 
        50, 
        10, 
        min_sigma=1e-3, 
        big_sigma=1e1, 
        wait_iter=5
    )

    pool = mp.Pool() 

    LENGTH = 10000
    times = 0 

    try: 
        for i in range(1000): 
            scores = [] 
            pop = e.ask() 

            for ind in pop: 
                # scores.append(fitness_cartpole(ind, steps=LENGTH)) 
                scores.append(pool.apply_async(fitness_cartpole, ((ind, False, LENGTH)))) 

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

            ind = pop[np.argmax(scores)] 
            # print(ind) 
            fitness_cartpole(ind, render=True) 

            if max_score == LENGTH: 
                times += 1 
            else: 
                times = 0 

            if times == 5: 
                ind = pop[np.argmax(scores)] 
                print(ind) 
                break 

    finally: 
        pass 
    # except Exception as e: 
    #     print("Error while training:", e) 