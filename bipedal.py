import multiprocessing as mp 
import sys 

import numpy as np 
import atari_py 
import gym 

import comm 
import hyperneat 
import neat 
import nn 

for name in gym.envs.registry.all(): 
    print(name) 
sys.exit(0) 

env = gym.make('Pong-ram-v0') 

print(env.reset().shape) 
print(env.action_space) 

def fitness_env(weights, render: bool=False, steps=1000): 
    # net.set_weights(weights) 
    net = weights
    score = 0
    # env._max_episode_steps = steps
    
    TRIES = 1

    for _ in range(TRIES): 
        # env.seed(0)
        obs = env.reset() 
        # env.seed(0) 

        while True: 
            close = False

            if render: 
                close = not env.render()

            # obs = np.transpose(obs, (2, 0, 1))

            # action = net.predict(np.expand_dims(obs, 0))[0]
            action = net.predict(obs)
            action = np.argmax(action) 

            obs, reward, done, _ = env.step(env.action_space.sample())

            score += reward

            if done or close: 
                break

    if render: 
        print("Done - {}".format(score / TRIES)) 
        env.close() 

    # print("Done") 

    return score / TRIES

# x = i = nn.Input((128,)) 
# x = nn.Dense(6, activation='sigmoid')(x) 
# net = nn.Model(i, x) 

x = i = nn.Input((128,)) 
# x = nn.MaxPool2D()(x) 
# x = nn.Conv2D(4, 3, padding='valid')(x) 
# x = nn.MaxPool2D()(x) 
# x = nn.Conv2D(4, 3, padding='valid')(x) 
# x = nn.MaxPool2D()(x) 
# x = nn.Conv2D(4, 3, padding='valid')(x) 
# x = nn.MaxPool2D()(x) 
# x = nn.Conv2D(4, 3, padding='valid')(x) 
# x = nn.MaxPool2D()(x) 
# x = nn.Flatten()(x) 
# x = nn.Dense(16)(x) 
x = nn.Dense(6)(x) 
net = nn.Model(i, x) 

if __name__ == "__main__": 
    # args = {
    #     'n_pop': 500, 
    #     'max_species': 10, 
    #     'species_threshold': 2.5, 
    #     'clear_species': 15, 
    #     'prob_add_node': 0.9, 
    #     'prob_add_conn': 0.8, 
    #     'prob_replace_weight': 0.4, 
    #     'prob_mutate_weight': 0.9, 
    #     'prob_toggle_conn': 0.3, 
    #     'prob_replace_activation': 0.0, 
    #     'std_new': 10.0, 
    #     'std_mutate': 0.1, 
    #     # 'activations': ['sigmoid'] 
    # }
    # args = {
    #         'n_pop': 100, 
    #         'max_species': 100, 
    #         'species_threshold': 0.75, 
    #         'clear_species': 15, 
    #         'prob_add_node': 0.05, 
    #         'prob_add_conn': 0.1, 
    #         'prob_replace_weight': 0.05, 
    #         'prob_mutate_weight': 0.2, 
    #         'prob_toggle_conn': 0.2, 
    #         'prob_replace_activation': 0.2, 
    #         'std_new': 5.0, 
    #         'std_mutate': 0.1 
    # }
    neat_args = {
            'n_pop': 200, 
            'max_species': 30, 
            'species_threshold': 0.35, 
            'clear_species': 15, 
            'prob_add_node': 0.01, 
            'prob_add_conn': 0.1, 
            'prob_replace_weight': 0.01, 
            'prob_mutate_weight': 0.1, 
            'prob_toggle_conn': 0.2, 
            'prob_replace_activation': 0.1, 
            'std_new': 1.0, 
            'std_mutate': 0.1, 
            'activations': ['sigmoid'], 
            'dist_weight': 0.1 
    }

    # hn = hyperneat.HyperNeat(net.get_config(), args) 
    hn = neat.Neat(128, 6, neat_args) 

    filename_out = sys.argv[1]
    file_out = open(filename_out, "w+") 

    file_out_name = "neat-{}".format(neat_args['n_pop'])
    file_out.write("{}\n".format(file_out_name)) 

    scores = [] 
    pop = [] 

    MAX_STEPS = 5000
    last_good = False 
    cur_max = -10000000

    pool = mp.Pool(processes=12) 

    try: 
        for i in range(500000): 
            pop = hn.ask() 

            scores.clear() 
            for ind in pop: 
                # scores.append(fitness_env(ind.get_weights(), False, steps=MAX_STEPS))
                # scores.append(pool.apply_async(fitness_env, ((ind.get_weights(), False))))
                # scores.append(fitness_env(ind, False, steps=MAX_STEPS))
                scores.append(pool.apply_async(fitness_env, ((ind, False))))

            scores = [s.get() for s in scores] 

            hn.tell(scores) 

            if np.max(scores) >= MAX_STEPS: 
                if last_good: 
                    break 
                else: 
                    last_good = True 
            else: 
                last_good = False 

            # file_m = open("{}-{}.model".format(filename_out, hn.gen), "wb+") 
            # file_m.write(comm.encode({
            #     'config': pop[np.argmax(scores)].get_config(), 
            #     'weights': pop[np.argmax(scores)].get_weights() 
            # }))
            # file_m.close() 
            file_out.write("{}\n".format(np.max(scores)))
            if np.max(scores) > cur_max: 
            # if True: 
                # fitness_env(pop[np.argmax(scores)].get_weights(), True, steps=10000)
                fitness_env(pop[np.argmax(scores)], True, steps=10000)
                cur_max = np.max(scores) 

    except: 
        while True: 
            print("Best score:", np.max(scores))
            # print("Long score:", fitness_env(pop[np.argmax(scores)].get_weights(), True, steps=100000))
            print("Long score:", fitness_env(pop[np.argmax(scores)], True, steps=100000))

    file_out.close() 