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

            print("Generation {}/{}: best - {:0.3e}, best weights - {}".format(gen + 1, n_gen, fitnesses[inds_f[0]], parents[0]), end='') 

            x_prime = np.copy(x)
            x += lr * np.mean(parents - x_prime, axis=0)
            std = np.sqrt(np.mean(np.square(parents - x_prime), axis=0)) + 1e-9

            print(std) 

            time.sleep(0.1) 
    finally: 
        srv.stop() 
        # pass

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
    es("fitness_pi", np.full((2,), 0), 10.0, n_gen=500, n_pop=10000, n_select=1) 