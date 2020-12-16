import os 

import argparse 
import multiprocessing as mp
import threading
import time 

import distrib 
import es 

def task_pong_es(data={}, shared={}): 
    import pong_es_distrib 
    return pong_es_distrib.fitness_pong(data['w']) 

def task_pong_neat(data={}, shared={}): 
    import pong_neat_distrib 
    import neat 
    return pong_neat_distrib.fitness_pong(neat.Genome.load(data['w']))  

distrib.register_task('pong_es', task_pong_es)
distrib.register_task('pong_neat', task_pong_neat)

def cmd(): 
    while True: 
        x = input().lower() 
        if x == 'exit': 
            break 
        else: 
            print("Unknown command: {}".format(x)) 
        time.sleep(1) 

if __name__ == "__main__": 
    mp.set_start_method("spawn") 
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--threads", type=int, default=0)
    args.add_argument("-a", "--address", type=str, default="localhost")
    args.add_argument("-p", "--port", type=int, default=4919)
    args = args.parse_args() 

    worker = distrib.DistributedWorker(args.address, port=args.port, n_threads=args.threads) 
    worker.start() 

    cmd() 
    worker.stop() 
