import os 
# os.environ['OPENBLAS_NUM_THREADS'] = '1' 
# os.environ['MKL_NUM_THREADS'] = '1' 

import time 

import numpy as np 

import comm 
import distrib 
import nn 

class EvolutionStrategy: 

    def __init__(self, x0, sigma0, n_pop, n_select, min_sigma=1-5, big_sigma=1e-1, wait_iter=10, min_incr=1e-2): 
        self.x = np.array(x0, dtype=np.float) 
        self.sigma = np.full_like(x0, sigma0, dtype=np.float)  
        self.min_sigma = min_sigma
        self.big_sigma = big_sigma
        self.n_pop = n_pop 
        self.n_select = n_select 
        self.waiting = False 
        self.gen = 0 
        self.best_score = -float('inf') 
        self.best_iter = 0 
        self.iter = 0 
        self.wait_iter = wait_iter
        self.min_incr = min_incr

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

        self.iter += 1

        self.waiting = False 
        inds_f = np.argsort(scores)[::-1]
        parents = self._pop[inds_f[:self.n_select]]

        min_sigma = self.min_sigma 

        best = scores[inds_f[0]] 
        if best - self.best_score > self.min_incr: 
            self.best_score = best 
            self.best_iter = self.iter 
        else: 
            if self.iter - self.best_iter > self.wait_iter: 
                min_sigma = self.big_sigma
                self.best_iter = self.iter 
                print("Taking too long, varying weights")

        print("Generation {}: best - {:0.6f}, elite - {:0.6f}, avg - {:0.6f}, worst - {:0.6f}".format(self.gen, scores[inds_f[0]], np.mean([scores[i] for i in inds_f[:self.n_select]]), np.mean(scores), scores[inds_f[-1]])) 

        self.sigma[:] = np.sqrt(np.mean(np.square(parents - self.x), axis=0)) + min_sigma 
        self.x += np.mean(parents - self.x, axis=0)
