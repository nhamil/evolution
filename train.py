from connect4 import Connect4 
import neat 

import multiprocessing as mp 

import numpy as np 

def play_games(a: neat.Network, pop: list): 
    play = np.arange(len(pop))[:30] 

    total = 0.0 
    for i in play: 
        total += play_game(a, pop[i]) 

    return total 

def play_game(a: neat.Network, b: neat.Network, p: bool=False): 
    env = Connect4(7, 6) 

    testee_num = 1 
    testee = a 

    i = 0# env.width * env.height 
    mi = env.width * env.height

    while env.get_winner() is None: 
        i += 1 
        state = env.get_state() 
        actions = env.get_actions() 
        player = env.get_player() 

        net = a if player == 1 else b 
        states = np.repeat(state[np.newaxis, :,:], len(actions), axis=0)
        for i in range(len(actions)): 
            env.step(actions[i], states[i])
        states = np.reshape(states, (len(actions), -1)) 

        values = [] 
        if player == 1: 
            values = np.array([net.predict(states[i]) for i in range(len(states))])[:,0]
        else: 
            values = np.zeros(len(actions))
            values[np.random.randint(0, len(values))] = 1 
        
        try: 
            env.step(actions[np.argmax(values)]) 
            if p: env.render() 
        except: 
            # print("Error while making move: {}".format(e))
            if p: print("{} won the game due to bad move by opponent".format('Testee' if testee == net else 'Opponent'))
            if testee == net: 
                return -10 + i/mi
            else: 
                return 1 + i/mi

    win = env.get_winner()
    if win == 0: 
        if p: print("Players tied")
        return 0 + i/mi
    elif win == testee_num: 
        if p: print("Testee won the game")
        return 1 + i/mi
    else: 
        if p: print("Opponent won the game")
        return -1 + i/mi

if __name__ == "__main__":
    neat = neat.Neat(6 * 7, 1, {
        'n_pop': 100 
    })

    pop = []
    scores = [] 

    try: 
        pool = mp.Pool(processes=12) 

        for it in range(20): 
            pop = neat.ask() 

            scores = [
                pool.apply_async(play_games, args=(i, pop)) for i in pop 
            ]

            scores = [
                s.get() for s in scores
            ]

            neat.tell(scores) 
        
        inds = np.argsort(scores) 
        play_game(pop[inds[-1]], pop[inds[-2]], p=True)
    finally: 
        pool.close() 
        pool.terminate() 
        pool.join() 
