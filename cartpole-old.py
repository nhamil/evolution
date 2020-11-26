import sys 

import numpy as np 
import atari_py 
import gym 

import nn 
import hyperneat 

env = gym.make('CartPole-v0') 

def fitness_env(env, net: nn.Model, render: bool=False, steps=1000): 
    score = 0
    env._max_episode_steps = steps
    obs = env.reset() 

    while True: 
        close = False

        if render: 
            close = not env.render()

        res = net.predict(np.expand_dims(obs, 0))[0]
        action = np.argmax(res) 

        obs, reward, done, _ = env.step(action)

        score += reward

        if done or close: 
            break

    return score 

x = i = nn.Input((4,)) 
x = nn.Dense(2)(x) 
x = nn.Dense(2)(x) 
net = nn.Model(i, x) 

args = {
    'n_pop': 100, 
    'max_species': 20, 
    'species_threshold': 1.5, 
    'clear_species': 15, 
    'prob_add_node': 0.3, 
    'prob_replace_weight': 0.4, 
    'prob_mutate_weight': 0.9, 
    'prob_toggle_conn': 0.3, 
    'prob_replace_activation': 0.4, 
    'std_new': 10.0, 
    'std_mutate': 0.1 
}

hn = hyperneat.HyperNeat(net.get_config(), args) 

filename_out = sys.argv[1]
file_out = open(filename_out, "w+") 

file_out.write("hyperneat-{}\n".format(args['n_pop']))

scores = [] 
pop = [] 

MAX_STEPS = 5000
last_good = False 

for i in range(500): 
    pop = hn.ask() 

    scores.clear() 
    for ind in pop: 
        scores.append(fitness_env(env, ind, False, steps=MAX_STEPS))

    hn.tell(scores) 

    if np.max(scores) >= MAX_STEPS: 
        if last_good: 
            break 
        else: 
            last_good = True 
    else: 
        last_good = False 

    file_out.write("{}\n".format(np.max(scores)))

print("Best score:", np.max(scores))
print("Long score:", fitness_env(env, pop[np.argmax(scores)], True, steps=100000))

file_out.close() 