# import os 
# os.environ['OPENBLAS_NUM_THREADS'] = '1' 
# os.environ['MKL_NUM_THREADS'] = '1' 

import time 

import numpy as np 

import distrib 

def fitness_pi(data={}, shared={}): 
    x = data['x']
    out = np.array(np.square(np.pi - x[0]) + np.square(1.414 - x[1])) 
    return out 

distrib.register_task("fitness_pi", fitness_pi) 

def es(): 
    srv = distrib.DistributedServer() 
    
    try: 
        srv.start() 
        time.sleep(5)

        POP_SIZE = 300 
        SELECT_SIZE = 5 
        MAX_GENERATION = 30
        LEARNING_RATE = 1 

        fit_fn = "fitness_pi" 

        x0 = [0.0, 0.0] 
        std0 = 10

        x = np.array(x0) 
        std = np.full_like(x0, std0) 

        for gen in range(MAX_GENERATION): 
            pop = np.random.randn(POP_SIZE, *x.shape) * std + x

            fit = []
            for x in pop: 
                fit.append(srv.execute(fit_fn, { 'x': x }))
            fitnesses = np.array([f.result() for f in fit])
            inds_f = np.argsort(fitnesses)
            parents = pop[inds_f[:SELECT_SIZE]]

            print("Generation {}/{}: best - {:0.3e}, best weights - {}".format(gen + 1, MAX_GENERATION, fitnesses[inds_f[0]], parents[0])) 

            x_prime = np.copy(x)
            x += LEARNING_RATE * np.mean(parents - x_prime, axis=0)
            std = np.sqrt(np.mean(np.square(parents - x_prime), axis=0))
    finally: 
        srv.stop() 

# def cmaes(): 
#     n_pop = 10 
#     n_sel = 5 
#     n_gen = 30 
#     fit_fn = fitness_pi 

#     alpha_mu = 1.0 

#     mu0 = [[0.0, 0.0]] 
#     sigma0 = 1 

#     mu = np.array(mu0) 
#     sigma = np.full_like(mu, sigma0) 

#     for gen in range(n_gen): 
#         pop = np.random.randn(n_pop, *mu.shape) * sigma + mu 

#         fit = fit_fn(pop) 
#         i_sort = np.argsort(fit) 
#         sel = pop[i_sort[:n_sel]] 

#         print("Generation {}/{}: best - {:0.3e}: {}".format(gen + 1, n_gen, fit[i_sort[0]], sel[0])) 

#         mu_prime = mu
#         mu = mu_prime + alpha_mu * np.mean(sel - mu_prime, axis=0) 

if __name__ == "__main__": 
    es()