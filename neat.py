import numpy as np 

class Genome: 

    def __init__(self, n_inputs, n_outputs, nodes: list, conns: list): 
        self.n_inputs = n_inputs 
        self.n_outputs = n_outputs 
        self.nodes = nodes # [(id, layer, activation, bias), ...]
        self.conns = conns # [(innov, enabled, a, b, weight), ...]
        self.nodes.sort(key=lambda x: x[0]) # sort by id 
        self.conns.sort(key=lambda x: x[0]) # sort by innovation number 

    def __str__(self): 
        return "Genome[{}, {}, {}, {}]".format(self.n_inputs, self.n_outputs, self.nodes, self.conns) 

    def copy(self): 
        return Genome(
            self.n_inputs, 
            self.n_outputs, 
            [x for x in self.nodes], 
            [x for x in self.conns] 
        )

    def layer(self, id): 
        if id < self.n_inputs: 
            return 0 
        elif id < self.n_outputs: 
            return 1 
        else: 
            index = self.find_node(id) 
            if index is not None: 
                return self.nodes[index][1] 
            else: 
                raise Exception("Node is not in genome: {}".format(id))
    
    def find_node(self, id): 
        lo = 0 
        hi = len(self.nodes) 
        while lo <= hi: 
            md = (lo + hi) // 2 
            m = self.nodes[md][0]
            if m == id: 
                return md 
            elif m < id: 
                lo = md + 1 
            else: 
                hi = md - 1 
        return None 

    def find_conn(self, a, b): 
        for i in range(len(self.conns)): 
            c = self.conns[i] 
            if c[2] == a and c[3] == b: 
                return i 
        return None 

    @staticmethod 
    def from_config(cfg): 
        return Genome(cfg['inputs'], cfg['outputs'], cfg['nodes'], cfg['conns'])

    def get_config(self): 
        return {
            'inputs': self.n_inputs, 
            'outputs': self.n_outputs, 
            'nodes': self.nodes, 
            'conns': self.conns 
        }  

    @property
    def max_innov(self): 
        return self.conns[-1][0] 

    @property 
    def max_node(self): 
        return len(self.nodes) 

class Network: 

    def __init__(self, genome: Genome): 
        self.genome = genome 
        self.n_inputs = genome.n_inputs
        self.n_outputs = genome.n_outputs 
        raise NotImplementedError 

    def predict(self, x): 
        raise NotImplementedError 

class Neat: 

    def __init__(self, n_inputs, n_outputs, cfg: dict={}): 
        if n_inputs < 1: 
            raise Exception("Number of inputs must be at least 1: {}".format(n_inputs))
        if n_outputs < 1: 
            raise Exception("Number of outputs must be at least 1: {}".format(n_outputs))
        self.n_inputs = n_inputs 
        self.n_outputs = n_outputs 
        self.node = n_inputs + n_outputs - 1
        self.innov = n_inputs * n_outputs - 1

    def next_innov(self): 
        self.innov += 1 
        return self.innov 

    def next_node(self): 
        self.node += 1 
        return self.node 

    def ask(self): 
        raise NotImplementedError

    def tell(self, scores: list): 
        raise NotImplementedError 

    def create_genome(self): 
        nodes = [] 
        conns = [] 

        for i in range(self.n_outputs): 
            nodes.append((self.n_inputs + i, 1.0, 'sigmoid', np.random.randn() * 1.0))
            for j in range(self.n_inputs): 
                conns.append(((i * self.n_inputs + j), True, j, (self.n_inputs + i), np.random.randn() * 1.0))

        return Genome(self.n_inputs, self.n_outputs, nodes, conns) 

    def distance(self, a: Genome, b: Genome, conn_d: float=1.0, conn_w: float=1.0, node_d: float=1.0, node_w: float=1.0, node_a: float=1.0): 
        disjoint_conns = 0 
        weights_conns = 0
        disjoint_nodes = 0 
        weights_nodes = 0 
        act_nodes = 0 

        N_conns = max(len(a.conns), len(b.conns)) 
        N_nodes = max(len(a.nodes), len(b.nodes)) 

        ai = 0 
        bi = 0 
        while ai < len(a.conns) and bi < len(b.conns): 
            if a.conns[ai][0] == b.conns[bi][0]: 
                # same innovation 
                weights_conns += abs(a.conns[ai][-1] - b.conns[bi][-1]) 
                ai += 1 
                bi += 1 
            elif a.conns[ai][0] < b.conns[bi][0]: 
                # add a's connection
                disjoint_conns += 1 
                ai += 1 
            else: 
                # add b's connection 
                disjoint_conns += 1 
                bi += 1 
        disjoint_conns += len(a.conns) - ai 
        disjoint_conns += len(b.conns) - bi 

        ai = 0 
        bi = 0 
        while ai < len(a.nodes) and bi < len(b.nodes): 
            if a.nodes[ai][0] == b.nodes[bi][0]: 
                # same innovation 
                weights_nodes += abs(a.nodes[ai][-1] - b.nodes[bi][-1]) 
                act_nodes += 0 if a.nodes[ai][-2] == b.nodes[bi][-2] else 1 
                ai += 1 
                bi += 1 
            elif a.nodes[ai][0] < b.nodes[bi][0]: 
                # add a's connection
                disjoint_nodes += 1 
                ai += 1 
            else: 
                # add b's connection 
                disjoint_nodes += 1 
                bi += 1 
        disjoint_nodes += len(a.nodes) - ai 
        disjoint_nodes += len(b.nodes) - bi 

        dist_conns = conn_d * disjoint_conns / N_conns + conn_w * weights_conns 
        dist_nodes = node_d * disjoint_nodes / N_nodes + node_w * weights_nodes + node_a * act_nodes

        return dist_conns + dist_nodes

    def crossover(self, a: Genome, b: Genome): 
        if a.n_inputs != b.n_inputs: 
            raise Exception("Number of inputs must be the same") 
        if a.n_outputs != b.n_outputs: 
            raise Exception("Number of outputs must be the same") 

        nodes = [] 
        conns = [] 

        # connections 
        ai = 0 
        bi = 0 
        while ai < len(a.conns) and bi < len(b.conns): 
            if a.conns[ai][0] == b.conns[bi][0]: 
                # same innovation 
                conns.append(a.conns[ai]) 
                ai += 1 
                bi += 1 
            elif a.conns[ai][0] < b.conns[bi][0]: 
                # add a's connection
                conns.append(a.conns[ai]) 
                ai += 1 
            else: 
                # add b's connection 
                conns.append(b.conns[bi]) 
                bi += 1 
        # excess connections 
        while ai < len(a.conns): 
            conns.append(a.conns[ai]) 
            ai += 1 
        while bi < len(b.conns): 
            conns.append(b.conns[bi]) 
            bi += 1 

        # nodes  
        ai = 0 
        bi = 0 
        while ai < len(a.nodes) and bi < len(b.nodes): 
            if a.nodes[ai][0] == b.nodes[bi][0]: 
                # same innovation 
                nodes.append(a.nodes[ai]) 
                ai += 1 
                bi += 1 
            elif a.nodes[ai][0] < b.nodes[bi][0]: 
                # add a's node
                nodes.append(a.nodes[ai]) 
                ai += 1 
            else: 
                # add b's node 
                nodes.append(b.nodes[bi]) 
                bi += 1 
        # excess nodes 
        while ai < len(a.nodes): 
            nodes.append(a.nodes[ai]) 
            ai += 1 
        while bi < len(b.nodes): 
            nodes.append(b.nodes[bi]) 
            bi += 1 

        return Genome(a.n_inputs, a.n_outputs, nodes, conns) 

    def mutate(self, g: Genome, add_node: float=0.1, add_conn: float=0.1, toggle_conn: float=0.1, nudge_weight: float=0.5, change_weight: float=0.05, nudge_strength: float=0.1, change_strength: float=1.0): 
        if np.random.random() < add_node: 
            a = np.random.randint(0, len(g.nodes) + self.n_inputs)
            if a >= self.n_inputs: 
                a = g.nodes[a - self.n_inputs][0] 
            b = g.nodes[np.random.randint(0, len(g.nodes))][0] 
            a_layer = g.layer(a) 
            b_layer = g.layer(b) 

            create = True 
            conn = None 

            # cannot add connection on same layer 
            if a_layer == b_layer: 
                create = False 
            # no recurrent 
            if a_layer > b_layer: 
                a_layer, b_layer = b_layer, a_layer 
                a, b = b, a 
            # disable direct connection if it exists 
            conn_ind = g.find_conn(a, b)
            if conn_ind is not None: 
                conn = g.conns[conn_ind]
                g.conns[conn_ind] = (conn[0], False, *conn[2:])

            if create: 
                # node id will be largest yet so appending will keep list sorted 
                layer = (a_layer + b_layer) * 0.5 
                node = self.next_node(), layer, 'sigmoid', np.random.randn() * change_strength 
                g.nodes.append(node) 

                conn_a = self.next_innov(), True, a, node[0], np.random.randn() * change_strength 
                conn_b = self.next_innov(), True, node[0], b, np.random.randn() * change_strength 

                # conn ids will be largest yet so appending will keep list sorted 
                g.conns.append(conn_a) 
                g.conns.append(conn_b) 

        if np.random.random() < add_conn: 
            a = np.random.randint(0, len(g.nodes) + self.n_inputs)
            if a >= self.n_inputs: 
                a = g.nodes[a - self.n_inputs][0] 
            b = g.nodes[np.random.randint(0, len(g.nodes))][0] 
            a_layer = g.layer(a) 
            b_layer = g.layer(b) 

            create = True 

            # cannot add connection on same layer 
            if a_layer == b_layer: 
                create = False 
            # no recurrent 
            if a_layer > b_layer: 
                a_layer, b_layer = b_layer, a_layer 
                a, b = b, a 
            # check if exists 
            if g.find_conn(a, b) is not None: 
                create = False 

            if create: 
                # conn id will be largest yet so appending will keep list sorted 
                conn = self.next_innov(), True, a, b, np.random.randn() * change_strength 
                g.conns.append(conn) 

        eps = 1e-2

        for i in range(len(g.nodes)): 
            node = g.nodes[i] 
            if np.random.random() < change_weight: 
                node = (*node[:-1], node[-1] + np.random.randn() * change_strength * (node[-1] + eps))
            if np.random.random() < nudge_weight: 
                node = (*node[:-1], node[-1] + np.random.randn() * nudge_strength * (node[-1] + eps))
            g.nodes[i] = node 

        for i in range(len(g.conns)): 
            conn = g.conns[i] 
            if np.random.random() < change_weight: 
                conn = (*conn[:-1], conn[-1] + np.random.randn() * change_strength * (conn[-1] + eps))
            if np.random.random() < nudge_weight: 
                conn = (*conn[:-1], conn[-1] + np.random.randn() * nudge_strength * (conn[-1] + eps))
            if np.random.random() < toggle_conn: 
                conn = (conn[0], not conn[1], *conn[2:])
            g.conns[i] = conn 

        return g 

if __name__ == "__main__": 
    neat = Neat(2, 1) 

    a = neat.create_genome() 
    b = neat.create_genome() 
    print(a)  
    print(b) 
    print(neat.distance(a, b)) 
    print() 
    c = neat.crossover(a, b) 
    print(c) 
    print() 
    neat.mutate(c, add_conn=1, add_node=1)
    print(c) 
    print() 
    print(c.get_config()) 