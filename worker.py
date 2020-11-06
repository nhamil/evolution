import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse 
import multiprocessing as mp
import threading
import time 

import distrib 
import es 

def cmd(): 
    while True: 
        x = input().lower() 
        if x == 'exit': 
            break 
        else: 
            print("Unknown command: {}".format(x)) 
        time.sleep(1) 

class MyPool: 

    def __init__(self, n_threads): 
        self.n_threads = n_threads 
        self.queue = mp.Queue() 
        self.procs = [mp.Process(target=self._run, args=(self.queue,)) for _ in range(n_threads)] 
        for proc in self.procs: 
            proc.start() 

    def _run(self, queue): 
        print("run") 
        pass 

    def execute(self, fn, args=None): 
        self.queue.put((fn, args)) 

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
