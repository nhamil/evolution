import numpy as np 

class Genome: 

    def __init__(self, n_inputs, n_outputs, nodes: list, conns: list): 
        self.n_inputs = n_inputs 
        self.n_outputs = n_outputs 
        self.nodes = nodes # [(id, layer, activation, bias), ...]
        self.conns = conns # [(innov, enabled, a, b, weight), ...]
        self.nodes.sort(key=lambda x: x[0]) # sort by id 
        self.conns.sort(key=lambda x: x[0]) # sort by innovation number 
        self.species = None 
        self.fitness = None 
        self.adj_fitness = None 

    def __str__(self): 
        return "Genome[{}, {}, {}, {}]".format(self.n_inputs, self.n_outputs, [(*n[:-1], float("{:0.3f}".format(n[-1]))) for n in self.nodes], [(*c[:-1], float("{:0.3f}".format(c[-1]))) for c in self.conns]) 

    def add_conn(self, conn): 
        self.conns.append(conn) 
        self.conns.sort(key=lambda x: x[0]) # sort by innovation number 

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

class Node: 

    def __init__(self, id, activation, act_name, bias): 
        self.id = id 
        self.conns = [] 
        self.activation = activation 
        self.act_name = act_name
        self.bias = bias 
        self.output = 0 

    def add_conn(self, in_node, weight): 
        self.conns.append((in_node, weight)) 

    def evaluate(self): 
        total = self.bias 
        for in_node, weight in self.conns: 
            total += in_node.output * weight 
        self.output = self.activation(total) 
        return self.output 

class Network: 

    def __init__(self, genome: Genome, activations: dict): 
        self.genome = genome 
        self.n_inputs = genome.n_inputs
        self.n_outputs = genome.n_outputs 

        layers = {}
        nodes = {} 

        self.input_layer = [] 
        for i in range(self.n_inputs): 
            node = Node(i, None, None, 0)
            self.input_layer.append(node) 
            nodes[i] = node 

        for n in genome.nodes: 
            layer = n[1] 
            if layer not in layers: 
                layers[layer] = [] 
            
            node = Node(n[0], activations[n[2]], n[2], n[3]) 
            nodes[n[0]] = node 
            layers[layer].append(node) 

        self.output_layer = [] 
        for i in range(self.n_outputs): 
            self.output_layer.append(nodes[self.n_inputs + i]) 

        for c in genome.conns: 
            enabled = c[1] 
            if enabled: 
                a = c[2] 
                b = c[3] 
                w = c[4] 
                nodes[b].add_conn(nodes[a], w) 

        keys = sorted([k for k in layers]) 
        self.layers = [] 
        for k in keys: 
            self.layers.append(layers[k]) 

    def __str__(self): 
        s = 'Network[in={}, out={}, species={}, nodes=[\n'.format(self.n_inputs, self.n_outputs, self.genome.species)
        for i in range(len(self.layers)): 
            layer = self.layers[i]
            for node in layer: 
                s += '  Node[id={}, layer={}, bias={:0.3f}, act={}, conns=['.format(node.id, i+1, node.bias, node.act_name)
                for conn in node.conns: 
                    if conn != node.conns[0]: 
                        s += ', '
                    s += '({}, {:0.3f})'.format(conn[0].id, conn[1]) 
                s += ']\n'
        s += ']'
        return s 

    def predict(self, x): 
        for i in range(len(self.input_layer)):       
            node = self.input_layer[i]       
            node.output = x[i]

        for layer in self.layers: 
            for node in layer: 
                node.evaluate() 
        
        return np.array([n.output for n in self.output_layer]) 

class Neat: 

    def __init__(self, n_inputs: int, n_outputs: int, n_pop: int=100, cfg: dict={}): 
        if n_inputs < 1: 
            raise Exception("Number of inputs must be at least 1: {}".format(n_inputs))
        if n_outputs < 1: 
            raise Exception("Number of outputs must be at least 1: {}".format(n_outputs))
        self.n_inputs = n_inputs 
        self.n_outputs = n_outputs 
        self.n_pop = n_pop 
        self.node = n_inputs + n_outputs - 1
        self.innov = n_inputs * n_outputs - 1
        self.innov_map = {} 
        self.spec = 0 
        self.species = {} 
        self.spec_sizes = None 
        self.generation = 0 
        self.elite_percent = 0.5 
        self.pop = None 
        self.activations = {
            'linear': lambda x: x, 
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)) 
        }

    def _next_innov(self): 
        self.innov += 1 
        return self.innov 

    def get_innov(self, a, b): 
        key = (a, b) 
        if key in self.innov_map: 
            return self.innov_map[key] 
        else: 
            i = self._next_innov() 
            self.innov_map[key] = i 
            return i 

    def next_node(self): 
        self.node += 1 
        return self.node 

    def next_species(self): 
        self.spec += 1 
        return self.spec 

    def ask(self): 
        self.generation += 1 

        if self.spec_sizes is None: 
            self.pop = [self.create_genome() for _ in range(self.n_pop)] 
        else: 
            self.pop = [] 
            for s in self.spec_sizes: 
                sz = self.spec_sizes[s] 
                spec = self.species[s] 
                spec.sort(key=lambda x: -x.fitness)
                elite = spec[:max(1, int(self.elite_percent * len(spec)))]
                # print("Creating {} individuals from {} elite of a species of {}".format(sz, len(elite), len(spec)))
                for _ in range(sz): 
                    a = None 
                    b = None 
                    if np.random.random() < 0.1: 
                        a = spec[np.random.randint(0, len(spec))] 
                    else: 
                        a = elite[np.random.randint(0, len(elite))] 

                    if np.random.random() < 0.1: 
                        b = spec[np.random.randint(0, len(spec))] 
                    else: 
                        b = elite[np.random.randint(0, len(elite))] 

                    c = self.crossover(a, b) 
                    self.mutate(c) 
                    self.pop.append(c) 

        last = self.species
        self.species = {} 

        for g in self.pop: 
            if g.species is not None: 
                self._add_genome_to_species_list(g) 

        species_threshold = 2.0

        for g in self.pop: 
            if g.species is None: 
                found = False 
                for s in last: 
                    lst = last[s]
                    compare = lst[np.random.randint(0, len(lst))] 
                    dist = self.distance(g, compare) 

                    # print("Checking distance to species {}: {}".format(s, dist))

                    if dist < species_threshold: 
                        g.species = s 
                        self._add_genome_to_species_list(g) 
                        found = True 
                        break 
                
                if not found: 
                    g.species = self.next_species() 
                    self.species[g.species] = [g] 
                    last[g.species] = [g] 

        return [self.create_network(g) for g in self.pop] 

    def tell(self, scores: list): 
        inds = np.argsort(scores)[::-1]

        for i in range(self.n_pop): 
            self.pop[i].fitness = scores[i] 

        mean_fit = 0
        spec_fit = { s: 0 for s in self.species } 
        for g in self.pop: 
            g.adj_fitness = g.fitness / len(self.species[g.species]) 
            spec_fit[g.species] += g.adj_fitness 
            mean_fit += g.adj_fitness 
        mean_fit /= self.n_pop 
        
        # print("Current species: ") 
        # self._print_species() 

        # print("New species sizes: ")
        self.spec_sizes = {} 
        total_size = 0 
        for s in self.species: 
            sz = int(spec_fit[s] / mean_fit)
            total_size += sz 
            self.spec_sizes[s] = sz
            # print("{}".format(sz))
        # print("Total size: {}".format(total_size))

        spec_keys = [s for s in self.species]

        while total_size < self.n_pop: 
            self.spec_sizes[spec_keys[np.random.randint(0, len(spec_keys))]] += 1
            total_size += 1

        while total_size > self.n_pop: 
            s = spec_keys[np.random.randint(0, len(spec_keys))]
            self.spec_sizes[s] -= 1
            if self.spec_sizes[s] <= 0: 
                del self.spec_sizes[s] 
                spec_keys.remove(s) 
            total_size -= 1

        # print("Corrected sizes: ")
        # for s in self.species: 
        #     print("{}".format(self.spec_sizes[s]))
        # print("Total size: {}".format(total_size))

        print("Generation {}: species: {}, best - {:0.3f}, avg - {:0.3f}".format(self.generation, len(self.species), scores[inds[0]], np.mean(scores)))

        net = self.create_network(self.pop[inds[0]])
        # print("Best genome:  {}".format(net.genome))
        print("Best network: {}".format(net))
        print("{}".format(net.predict([0, 0])))
        print("{}".format(net.predict([0, 1])))
        print("{}".format(net.predict([1, 0])))
        print("{}".format(net.predict([1, 1])))

    def _add_genome_to_species_list(self, genome: Genome): 
        if genome.species is None: 
            raise Exception("Genome must have species before adding it to a species list") 
        if genome.species not in self.species: 
            self.species[genome.species] = [genome] 
        else: 
            self.species[genome.species].append(genome) 

    def _print_species(self): 
        for s in self.species: 
            print("{}: {} individuals".format(s, len(self.species[s])))#, self.species[s][0]))

    def create_network(self, genome: Genome): 
        return Network(genome, self.activations) 

    def create_genome(self): 
        nodes = [] 
        conns = [] 

        for i in range(self.n_outputs): 
            nodes.append((self.n_inputs + i, 1.0, 'sigmoid', np.random.randn() * 1.0))
            for j in range(self.n_inputs): 
                conns.append(((i * self.n_inputs + j), True, j, (self.n_inputs + i), np.random.randn() * 1.0))

        return Genome(self.n_inputs, self.n_outputs, nodes, conns) 

    def distance(self, a: Genome, b: Genome, conn_d: float=0.1, conn_w: float=0.1, node_d: float=0.5, node_w: float=0.1, node_a: float=1.0): 
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

    def mutate(self, g: Genome, add_node: float=0.1, add_conn: float=1.0, toggle_conn: float=0.0, nudge_weight: float=0.5, change_weight: float=0.1, nudge_strength: float=0.1, change_strength: float=1.0): 
        g.fitness = None 
        g.adj_fitness = None 
        g.species = None 
        
        if np.random.random() < add_node: 
            a = np.random.randint(0, len(g.nodes) + self.n_inputs)
            if a >= self.n_inputs: 
                a = g.nodes[a - self.n_inputs][0] 
            b = g.nodes[np.random.randint(0, len(g.nodes))][0] 
            a_layer = g.layer(a) 
            b_layer = g.layer(b) 

            create = True 
            conn = None 

            # TODO remove this
            if a_layer != 0 or b_layer != 1: 
                create = False

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

                conn_a = self.get_innov(a, node[0]), True, a, node[0], np.random.randn() * change_strength 
                conn_b = self.get_innov(node[0], b), True, node[0], b, np.random.randn() * change_strength 

                # conn ids will be largest yet so appending will keep list sorted 
                g.add_conn(conn_a) 
                g.add_conn(conn_b) 

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
            e = g.find_conn(a, b)
            if e is not None: 
                create = False 

            if create: 
                conn = self.get_innov(a, b), True, a, b, np.random.randn() * change_strength 
                g.add_conn(conn) 

        eps = 1e-2

        for i in range(len(g.nodes)): 
            node = g.nodes[i] 
            if np.random.random() < change_weight: 
                node = (*node[:-1], np.random.randn() * change_strength)# * (node[-1] + eps))
            if np.random.random() < nudge_weight: 
                node = (*node[:-1], node[-1] + np.random.randn() * nudge_strength)# * (node[-1] + eps))
            g.nodes[i] = node 

        for i in range(len(g.conns)): 
            conn = g.conns[i] 
            if np.random.random() < change_weight: 
                conn = (*conn[:-1], np.random.randn() * change_strength)# * (conn[-1] + eps))
            if np.random.random() < nudge_weight: 
                conn = (*conn[:-1], conn[-1] + np.random.randn() * nudge_strength)# * (conn[-1] + eps))
            if np.random.random() < toggle_conn: 
                conn = (conn[0], not conn[1], *conn[2:])
            g.conns[i] = conn 

        return g 

if __name__ == "__main__": 
    neat = Neat(2, 1, n_pop=100) 

    # g = Genome(2, 1, [
    #     (2, 1.0, 'sigmoid', -30), 
    #     (3, 0.5, 'sigmoid', -10), 
    #     (4, 0.5, 'sigmoid', 30) 
    # ], [
    #     (1, True, 0, 3, 20.0), 
    #     (2, True, 1, 3, 20.0), 
    #     (3, True, 0, 4, -20.0), 
    #     (4, True, 1, 4, -20.0),
    #     (5, True, 3, 2, 20.0), 
    #     (6, True, 4, 2, 20.0)
    # ])

    # nn = neat.create_network(g) 
    # print(nn) 
    # print(nn.predict([0, 0])) 
    # print(nn.predict([0, 1])) 
    # print(nn.predict([1, 0])) 
    # print(nn.predict([1, 1])) 

    # f = 0 
    # p = 1.0
    # f += np.power(np.abs(nn.predict([0, 0]) - 1), p) 
    # f += np.power(np.abs(nn.predict([0, 1]) - 0), p) 
    # f += np.power(np.abs(nn.predict([1, 0]) - 0), p) 
    # f += np.power(np.abs(nn.predict([1, 1]) - 1), p) 
    # f = np.sum(f) 
    # print(f) 

    for _ in range(10000): 
        pop = neat.ask() 
        fit = [] 

        for nn in pop: 
            f = 0 
            p = 2.0
            f += np.power(np.abs(nn.predict([0, 0]) - 1), p) 
            f += np.power(np.abs(nn.predict([0, 1]) - 0), p) 
            f += np.power(np.abs(nn.predict([1, 0]) - 0), p) 
            f += np.power(np.abs(nn.predict([1, 1]) - 1), p) 
            f = np.sum(f) 
            fit.append(f) 

        neat.tell(fit) 