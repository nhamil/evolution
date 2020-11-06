import multiprocessing as mp 
import random 
import select 
import struct 
import socket 
import sys 
import threading 
import time 
import traceback
import queue 

import psutil 

import comm 

MSG_SERVER_DISCONNECT = 0x01 
MSG_CLIENT_DISCONNECT = 0x02
MSG_SERVER_TASK = 0x03 
MSG_CLIENT_RESULT = 0x04 
MSG_SERVER_MESSAGE = 0x05 
MSG_CLIENT_CONNECT = 0x06 
MSG_SERVER_GET_INFO = 0x07 
MSG_CLIENT_SEND_INFO = 0x08 
MSG_SERVER_KICK = 0x09 
MSG_SERVER_SHARE = 0x0A 

VERSION = 17

TASKS = {}

def register_task(name, func): 
    TASKS[name] = func 

def get_task(name): 
    return TASKS[name] 

def task_ping(data={}, shared={}): 
    return "Pong!" 

def task_rand_wait(data={}, shared={}): 
    wait = random.uniform(0.1, 1.0) 
    time.sleep(wait) 
    return "Waited {} seconds".format(wait) 

def task_add_indirect(data={}, shared={}): 
    print("doing add indirect task: {}, {}".format(data, shared))  
    res = shared[data['a']] + shared[data['b']] 
    print("done") 
    return res 

register_task("ping", task_ping) 
register_task("wait", task_rand_wait) 
register_task("add_indirect", task_add_indirect)

class DistributedFuture: 

    def __init__(self, server, id): 
        self._server = server 
        self._id = id 
        self._result = None 
        self._done = False 
        self._lock = threading.Lock() 

    @property 
    def id(self): 
        return self._id 

    def done(self): 
        with self._lock: 
            return self._done 

    def result(self, timeout=None): 
        waited = 0
        while timeout is None or waited <= timeout: 
            if waited > 0: 
                time.sleep(0.01) 
                waited += 0.01 
            with self._lock: 
                if self._done: 
                    return self._result 
        return None 

    def _finish(self, result): 
        with self._lock: 
            self._result = result 
            self._done = True 

class DistributedServer: 

    def __init__(self, port: int=4919): 
        self.port = port 
        self.lock = threading.Lock() 
        self._running = False 
        self._workers = [] 
        self.server = None 
        self._next_id = 0 
        self._task_queue = queue.Queue() 
        self._cur_futures = [] 
        self._shared = {} 

    def next_id(self): 
        with self.lock: 
            self._next_id += 1 
            return self._next_id  

    @property 
    def is_running(self): 
        with self.lock: 
            return self._running 

    @property
    def n_total_threads(self): 
        sum = 0 
        with self.lock: 
            for worker in self._workers: 
                sum += worker.n_threads 
        return sum 

    @property
    def n_workers(self): 
        sum = 0 
        with self.lock: 
            for worker in self._workers: 
                if worker.is_ready: 
                    sum += 1
        return sum 

    @property
    def memory(self): 
        sum = 0 
        with self.lock: 
            for worker in self._workers: 
                if worker.is_ready: 
                    sum += worker._memory 
        return sum 

    @property
    def n_connected_workers(self): 
        with self.lock: 
            return len(self._workers)

    @property
    def n_queued_tasks(self): 
        return self._task_queue.qsize() 

    def start(self): 
        if self.is_running: 
            raise Exception("Pool is already running") 

        with self.lock: 
            self._running = True 

        self.server = threading.Thread(target=self._server_thread)
        self.server.start() 

    def broadcast(self, message: str): 
        with self.lock: 
            for worker in self._workers: 
                if worker.is_ready: 
                    try: 
                        comm.send_socket_message(worker.sock, MSG_SERVER_MESSAGE, message) 
                    except: 
                        print("Exception occurred while sending message to {}:{}".format(*worker.addr)) 

    def execute(self, task: str, args: dict={}): 
        future = DistributedFuture(self, self.next_id()) 
        self._task_queue.put((future, {"id": future.id, "task": task, "args": args})) 
        return future 

    def _ready(self): 
        for f in self._cur_futures: 
            if not f.done(): 
                return False 
        self._cur_futures = [] 
        return True 

    def _wait(self): 
        while not self._ready(): 
            time.sleep(0.01) 

    def ready(self): 
        with self.lock: 
            return self._ready() 

    def wait(self): 
        with self.lock: 
            self._wait() 

    def share(self, data): 
        with self.lock: 
            self._wait() 
            self._shared = data 
            for worker in self._workers: 
                if worker.is_ready: 
                    try: 
                        comm.send_socket_message(worker.sock, MSG_SERVER_SHARE, data) 
                    except: 
                        print("Exception occurred while sharing data with {}:{}".format(*worker.addr)) 

    def stop(self):  
        if not self.is_running: 
            raise Exception("Pool is not running")

        with self.lock: 
            self._running = False 

        try: 
            self.server.join() 
            self.server = None 
        except: 
            raise "Error while closing server" 

        workers = [x for x in self._workers]
        for client in workers: 
            try: 
                client.join() 
            except: 
                raise "Error while closing client" 
        self._workers = [] 

    def _server_thread(self): 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        print("Starting server at %s:%s" % ('', self.port))
        sock.bind(('', self.port))
        sock.listen(1) 

        while self.is_running: 
            readable, _, _ = select.select([sock], [], [], 0.1)
            try: 
                for s in readable: 
                    client, addr = s.accept() 
                    with self.lock: 
                        thread = DistributedServerWorkerThread(self, client, addr) 
                        self._workers.append(thread) 
                        thread.start() 
            except: 
                print("Error while waiting for client") 

            workers = []
            with self.lock: 
                for x in self._workers: 
                    if x.is_ready: 
                        for i in range(x._n_threads): 
                            workers.append(x) 
            random.shuffle(workers) 
            for w in workers: 
                if self._task_queue.empty(): 
                    break 

                if not w.can_take_task: 
                    continue 
            # while self._task_queue.qsize() > 0: 
            #     worker = None 
            #     random.shuffle(workers) 
            #     for w in workers: 
            #         if w.can_take_task: 
            #             worker = w 
            #             break 
            #     if worker is None: 
            #         break 

                future, send = self._task_queue.get() 
                w.add_task(future, send) 

        sock.close() 
        print("Stopping server at %s:%s" % ("localhost", self.port))

    def _remove_worker(self, worker): 
        with self.lock: 
            try: 
                self._workers.remove(worker) 
                with worker.lock: 
                    for id in worker._tasks: 
                        # print("Requeueing task {}".format(id)) 
                        self._task_queue.put(worker._tasks[id]) 
            except: 
                pass # worker was not found 

class DistributedServerWorkerThread(threading.Thread): 

    def __init__(self, server, sock, addr): 
        super().__init__() 

        self.server = server 
        self.sock = sock 
        self.addr = addr 
        self.lock = threading.Lock() 
        self._ready = False 
        self._n_threads = 0 
        self._n_tasks = 0 
        self._tasks = {} 

    @property 
    def is_ready(self): 
        with self.lock: 
            return self._ready 

    @property
    def n_threads(self): 
        with self.lock: 
            return self._n_threads 

    @property
    def n_tasks(self): 
        with self.lock: 
            return self._n_tasks 

    @property
    def can_take_task(self): 
        # with self.lock: 
        #     return self._n_threads * 3 > self._n_tasks 
        return True

    def add_task(self, future, data): 
        with self.lock: 
            # if self._n_threads * 3 > self._n_tasks: 
            self._n_tasks += 1 
            self._tasks[future.id] = (future, data)
            try: 
                comm.send_socket_message(self.sock, MSG_SERVER_TASK, data) 
            except: 
                # print("Failed to send task to worker: {} ({})".format(future.id, e))
                # print("Requeueing task {}".format(future.id)) 
                self.server._task_queue.put((future, data)) 
                del self._tasks[future.id] 
            # else: 
            #     raise Exception("Cannot take task") 

    def run(self): 
        try: 
            self.sock.settimeout(0.1) 

            while self.server.is_running: 
                if self.sock is not None and self.sock.fileno() > -1: 
                    try: 
                        readable, _, _ = select.select([self.sock], [], [], 0.1)
                        for s in readable: 
                            if s is not None and s.fileno() > -1: 
                                msg, data = comm.recv_socket_message(s) 
                                if msg is None: 
                                    print("Error occured while receiving message") 
                                    s.close() 
                                else: 
                                    self.recv_worker_message(s, self.addr, msg, data) 
                    except socket.timeout: 
                        pass
                else: 
                    break 
            
            if self.sock is not None and self.sock.fileno() > -1: 
                comm.send_socket_message(self.sock, MSG_SERVER_DISCONNECT, {"reason": "server shutting down"}) 

            print("Worker disconnected at {}:{}".format(*self.addr)) 
        except Exception as e: 
            print(e) 
        finally: 
            with self.lock: 
                self._ready = False 
                self._n_threads = 0 
            self.sock.close() 
            self.server._remove_worker(self) 

    def recv_worker_message(self, sock, addr, msg, data): 
        if msg == MSG_CLIENT_CONNECT: 
            try: 
                fail = None

                if data['version'] != VERSION: 
                    fail = "worker uses old version" 

                if type(data['n_threads']) != int: 
                    fail = "worker does not include thread count" 

                if data['n_threads'] <= 0: 
                    fail = "worker uses an invalid number of threads" 

                if type(data['memory']) != float: 
                    fail = "worker does not include available memory" 

                if data['memory'] <= 0: 
                    fail = "worker has an invalid amount of memory" 

                if fail is None: 
                    print("Worker connected at {}:{} with {} thread(s) and {:01.1f}GB of memory".format(*addr, data['n_threads'], data['memory'])) 
                else: 
                    if sock is not None and sock.fileno() > -1: 
                        comm.send_socket_message(sock, MSG_SERVER_KICK, { "reason": fail }) 
                        sock.close() 

                with self.lock: 
                    comm.send_socket_message(sock, MSG_SERVER_SHARE, self.server._shared) 
                    self._n_threads = data['n_threads'] 
                    self._memory = data['memory'] 
                    self._ready = True 
            except: 
                if sock is not None and sock.fileno() > -1: 
                    comm.send_socket_message(sock, MSG_SERVER_KICK, { "reason": "error while reading worker data" }) 
                    sock.close() 
        elif msg == MSG_CLIENT_DISCONNECT: 
            sock.close() 
        elif msg == MSG_CLIENT_RESULT: 
            try: 
                # print("Worker result: {}".format(data)) 
                future = None 
                with self.lock: 
                    if data['id'] in self._tasks: 
                        future = self._tasks[data['id']] 
                if future is not None: 
                    future[0]._finish(data['result']) 
                    with self.lock: 
                        del self._tasks[data['id']] 
                        self._n_tasks -= 1 
            except: 
                if sock is not None and sock.fileno() > -1: 
                    comm.send_socket_message(sock, MSG_SERVER_KICK, { "reason": "error while reading worker data" }) 
                    sock.close() 
        else: 
            print("Unknown message {}: {}".format(msg, data)) 

def _init(): 
    pass 

class DistributedWorker: 

    def __init__(self, host: str, port: int=4919, n_threads: int=0): 
        self.host = host 
        self.port = port 
        self.n_threads = n_threads if n_threads > 0 else mp.cpu_count() 
        self.pool = mp.Pool(self.n_threads)
        self.pool.apply(_init) 
        self._running = False 
        self.lock = threading.Lock() 
        self.worker = None 
        self.tasks = {} 
        self.shared = {} 

    @property 
    def is_running(self): 
        with self.lock: 
            return self._running 

    def start(self): 
        if self.is_running: 
            raise Exception("Worker is already running") 

        with self.lock: 
            self._running = True 

        print("Starting worker process with {} thread(s)".format(self.n_threads)) 

        self.worker = threading.Thread(target=self._worker_thread) 
        self.worker.start() 

    def stop(self):  
        if not self.is_running: 
            return "Worker is already running" 

        with self.lock: 
            self._running = False 

        print("Stopping worker") 

        self.pool.close() 
        self.pool.terminate() 
        self.pool.join() 

        self.worker.join() 
        self.worker = None 

    def _worker_thread(self): 
        pool = None
        wait = False 
        while self.is_running: 
            try: 
                if wait: 
                    time.sleep(1) 
                else: 
                    print("Connecting to {}:{}".format(self.host, self.port)) 

                wait = True 

                try: 
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                    sock.settimeout(1) 
                    try: 
                        sock.connect((self.host, self.port)) 
                        print("Connected!")
                    except: 
                        # print("Could not connect to server")
                        sock.close() 
                        sock = None 
                        continue  

                    pool = self.pool

                    mem = psutil.virtual_memory() 
                    starting_mem = mem.available / 1024 / 1024 / 1024

                    sock.settimeout(0.1) 
                    comm.send_socket_message(sock, MSG_CLIENT_CONNECT, {
                        "version": VERSION, 
                        "n_threads": self.n_threads, 
                        "memory": starting_mem 
                    }) 

                    while self.is_running: 
                        if sock.fileno() == -1: 
                            break 

                        finished = [] 
                        for id in self.tasks: 
                            task = self.tasks[id] 
                            if task.ready(): 
                                comm.send_socket_message(sock, MSG_CLIENT_RESULT, {"id": id, "result": task.get()}) 
                                finished.append(id) 
                        for id in finished: 
                            del self.tasks[id] 

                        try: 
                            msg, data = comm.recv_socket_message(sock) 
                            if msg is None: 
                                print("Error occured while receiving message") 
                                break 
                            else: 
                                if self._recv_message(sock, pool, msg, data): 
                                    break 
                        except socket.timeout: 
                            pass 

                except Exception as e: 
                    print("Exception occurred with server communication: {}".format(e)) 
                    traceback.print_exc() 

                finally: 
                    # if pool is not None: 
                    #     pool.close() 
                    #     pool.terminate() 
                    #     self.pool = mp.Pool(self.n_threads) 
                    if sock is not None: 
                        if sock.fileno() > -1: 
                            comm.send_socket_message(sock, MSG_CLIENT_DISCONNECT) 

                        if self.is_running: 
                            print("Disconnected from server, attempting to reconnect") 
                        else: 
                            print("Disconnected from server") 
                        sock.close()
                        sock = None 
                        self.tasks = {} 
                        self.shared = {} 
            except KeyboardInterrupt: 
                print("Keyboard interrupt: exitting") 
                with self.lock: 
                    self._running = False 
            except Exception as e: 
                print("Exception occurred ({}), restarting".format(e))
        
        print("Worker service has been shut down, type `exit` to quit application") 

    def _recv_message(self, sock, pool, msg, data): 
        if msg == MSG_SERVER_DISCONNECT: 
            print("Disconnecting from server: {}".format(data['reason'])) 
            sock.close() 
            return True 
        elif msg == MSG_SERVER_KICK: 
            print("Kicked from server: {}".format(data['reason'])) 
            sock.close() 
            with self.lock: 
                self._running = False 
            return True 
        elif msg == MSG_SERVER_MESSAGE: 
            print("Message from server: {}".format(data)) 
        elif msg == MSG_SERVER_TASK: 
            print("Task from server: {} (id={})".format(data['task'], data['id']))
            try: 
                task = get_task(data['task']) 
                self.tasks[data['id']] = pool.apply_async(task, (data['args'], self.shared))
            except Exception as e: 
                print(e) 
        elif msg == MSG_SERVER_SHARE: 
            print("New shared data from server") 
            self.shared = data 
        else: 
            print("Unknown message {}: {}".format(msg, data)) 
        return False 
