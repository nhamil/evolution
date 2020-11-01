import ast 
import multiprocessing as mp 
import select 
import socket 
import sys 
import threading 
import time 

import distrib

if __name__ == "__main__": 
    try: 
        server = distrib.DistributedServer() 
        server.start() 

        while True: 
            raw = input()
            cmd = raw.split() 
            if len(cmd) == 0: 
                continue 
            if cmd[0] == 'exit': 
                break 
            elif cmd[0] == 'info': 
                out = "\n" 
                out += "WORKER POOL INFO\n"
                out += "Worker count   : {}\n".format(server.n_workers) 
                out += "Worker threads : {}\n".format(server.n_total_threads) 
                out += "Total memory   : {:01.1f}GB\n".format(server.memory) 
                out += "Queued tasks   : {}\n".format(server.n_queued_tasks) 
                print(out) 
            elif cmd[0] == 'send': 
                msg = raw.split(maxsplit=1)[1]
                print("Sending message: {}".format(msg))
                server.broadcast(msg)
            elif cmd[0] == 'task' and len(cmd) >= 2: 
                num = 1 
                data = None
                num = int(cmd[2]) 
                if len(cmd) >= 4: 
                    data = ast.literal_eval(raw.split(maxsplit=3)[3])
                
                print("Requesting {} task(s): {}".format(num, cmd[1]))
                res = []
                for _ in range(num): 
                    res.append(server.execute(cmd[1], data)) 
                for r in res: 
                    r.result() 
                print("Done!") 
            elif cmd[0] == 'share': 
                data = ast.literal_eval(raw.split(maxsplit=1)[1])
                print("Sharing data: {}".format(data))
                server.share(data)
            else: 
                print("Unknown command: {}".format(raw)) 
    finally: 
        server.stop() 