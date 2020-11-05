# import os 
# os.environ['OPENBLAS_NUM_THREADS'] = '1' 
# os.environ['MKL_NUM_THREADS'] = '1' 

import time 

import numpy as np 

from connect4 import Connect4
import distrib 
import nn 

def fitness_pi(data={}, shared={}): 
    x = data['x']
    out = np.array(np.square(np.pi - x[0]) + np.square(1.414 - x[1])) 
    return out 

def fitness_xor(data={}, shared={}): 
    model = data['model'] 
    model = nn.dict_to_network(model) 

    out = model.predict(np.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1]
    ])).reshape((4,))

    if 'print' in data and data['print']: 
        for y in out: 
            print("{:0.3f}".format(y))

    out[0] = 1 - out[0] 
    out[3] = 1 - out[3] 

    out = np.power(out, 0.5) 

    return np.array(4 - np.sum(out)) 

def fitness_connect4(data={}, shared={}): 
    model_a = shared['models'][data['x']]
    model_a = nn.dict_to_network(model_a) 

    model_b = shared['models'][data['y']]
    model_b = nn.dict_to_network(model_b) 

    nets = { 1: model_a, -1: model_b }
    w, h = shared['size'] 

    env = Connect4(w, h) 

    p = False 
    if 'print' in data and data['print']: 
        p = True 

    while env.get_winner() == 0: 
        state = env.get_state() 

        player = env.get_player() 

        values = nets[player].predict(state.flatten()) 

        try: 
            env.step(int(np.argmax(values))) 
            if p: env.render() 
        except: 
            # print("Error while making move: {}".format(e))
            if p: print("{} won the game due to bad move by opponent".format('Y' if nets[player] == model_a else 'X'))
            if nets[player] == model_a: 
                return 0 
            else: 
                return 1 

    if p: print("{} won the game".format('X' if nets[env.get_winner()] == model_a else 'Y'))
    return 1 if nets[env.get_winner()] == model_a else 0 


distrib.register_task("fitness_pi", fitness_pi) 
distrib.register_task("fitness_xor", fitness_xor) 
distrib.register_task("fitness_connect4", fitness_connect4) 

def es(fit_fn, x0, std0=1, n_pop=100, n_select=5, n_gen=30, lr=1): 
    srv = distrib.DistributedServer() 
    
    print(fit_fn, x0, std0, n_pop, n_select, n_gen, lr)

    try: 
        srv.start() 
        time.sleep(5)

        x = np.array(x0) 
        std = np.full_like(x0, std0) 

        for gen in range(n_gen): 
            pop = np.random.randn(n_pop, *x.shape) * std + x

            fit = []
            for x in pop: 
                fit.append(srv.execute(fit_fn, { 'x': x }))
                # fit.append(fit_fn({ 'x': x }))
            fitnesses = np.array([f.result() for f in fit])
            # fitnesses = np.array(fit)
            inds_f = np.argsort(fitnesses)
            parents = pop[inds_f[:n_select]]

            print("Generation {}/{}: best - {:0.3e}".format(gen + 1, n_gen, fitnesses[inds_f[0]]), end='') 

            x_prime = np.copy(x)
            x += lr * np.mean(parents - x_prime, axis=0)
            std = np.sqrt(np.mean(np.square(parents - x_prime), axis=0)) + 1e-9

            print(std) 

            time.sleep(0.1) 
    finally: 
        srv.stop() 
        # pass

class EvolutionStrategy: 

    def __init__(self, x0, sigma0, n_pop, n_select): 
        self.x = np.array(x0, dtype=np.float) 
        self.sigma = np.full_like(x0, sigma0, dtype=np.float)  
        self.n_pop = n_pop 
        self.n_select = n_select 
        self.waiting = False 
        self.gen = 0 

    def ask(self): 
        if self.waiting: 
            raise Exception("A population has already been asked for") 

        self.waiting = True 
        self._pop = np.random.randn(self.n_pop, *self.x.shape) * self.sigma + self.x
        self.gen += 1
        return self._pop 

    def tell(self, scores): 
        if not self.waiting: 
            raise Exception("A population must be asked for") 

        self.waiting = False 
        inds_f = np.argsort(scores)
        parents = self._pop[inds_f[:self.n_select]]

        print("Generation {}: best - {:0.3e}".format(self.gen, scores[inds_f[0]])) 

        self.sigma = np.sqrt(np.mean(np.square(parents - self.x), axis=0)) + 1e-9
        self.x += np.mean(parents - self.x, axis=0)

# def cmaes(fit_fn, mu0, sigma0=1.0, n_pop=100, n_select=5, n_gen=30, lr=1): 
#     alpha_mu = lr 

#     mu = np.array(mu0, dtype=np.float) 
#     sigma = np.diag(np.full_like(mu.flatten(), sigma0, dtype=np.float))

#     print(mu.shape, sigma.shape)

#     for gen in range(n_gen): 
#         pop = np.reshape(np.random.multivariate_normal(mu.flatten(), sigma, size=n_pop), (n_pop, *mu.shape)) 
#     #     pop = np.random.randn(n_pop, *mu.shape) * sigma + mu 

#         fit = []
#         for x in pop: 
#             fit.append(fit_fn({ 'x': x }))
#         fitnesses = np.array([f for f in fit])
#         inds_f = np.argsort(fitnesses)
#         sel = pop[inds_f[:n_select]]

#         mu_prime = mu
#         mu = mu_prime + alpha_mu * np.mean(sel - mu_prime, axis=0) 

#         sel_2d = np.reshape(sel, (n_select, -1)) 
#         mu_1d = mu_prime.flatten()
#         # sel_2d += -np.mean(sel_2d, axis=0) + mu_1d
#         sel_2d -= mu_1d
#         # print(np.mean(sel_2d - np.mean(sel_2d, axis=0) + mu_1d, axis=0))
#         # print(mu_1d) 
#         sigma = np.cov(sel_2d.T)
#         # print(sigma) 

#         print("Generation {}/{}: best - {:0.3e}, best weights - {}, sigma - {}".format(gen + 1, n_gen, fitnesses[inds_f[0]], sel[0], sigma.flatten())) 

if __name__ == "__main__": 
    # es("fitness_pi", np.full((2,), 0), 10.0, n_gen=50, n_pop=100, n_select=1) 

    CON4_WIDTH = 7 
    CON4_HEIGHT = 6 

    base = nn.ModelBuilder(CON4_WIDTH * CON4_HEIGHT) 
    base.dense(100) 
    base.dense(CON4_WIDTH) 
    base = base.build() 

    w = base.get_weights() 

    es = EvolutionStrategy(w[0], 1.0, 50, 5) 

    srv = distrib.DistributedServer() 
    try: 
        srv.start() 
        time.sleep(5) 

        shared = {} 
        scores = [] 

        for i in range(10): 
            pop = es.ask() 

            shared = { 
                'models': [nn.network_to_dict(base, (x, w[1])) for x in pop], 
                'size': (CON4_WIDTH, CON4_HEIGHT) 
            }
            srv.share(shared) 

            scores = [] 
            for x in range(len(pop)): 
                # scores.append(srv.execute('fitness_xor', {'model': nn.network_to_dict(base, (x, w[1]))})) 
                s = [] 
                for y in range(len(pop)): 
                    if x != y: 
                        s.append(srv.execute('fitness_connect4', {'x': x, 'y': y}))
                scores.append(s) 

            scores = [-np.sum([s.result() for s in lst]) for lst in scores] 
            # scores = [s.result() for s in scores] 
            es.tell(scores) 

            # if sorted(scores)[0] < 0.1: 
            #     break 

        best = np.argsort(scores)
        print("Best 2 players: {} ({}), {} ({})".format(best[0], scores[best[0]], best[1], scores[best[1]]))

        # fitness_xor({'model': nn.network_to_dict(base, (es.x, w[1])), 'print': True})
        fitness_connect4(data={'x': best[0], 'y': best[1], 'print': True}, shared=shared)
    
    finally: 
        srv.stop()
        pass 