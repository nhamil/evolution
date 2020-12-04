import numpy as np 

class NodeGene: 

    def __init__(self, id: int, layer: float, activation: str, bias: float): 
        self.id = id 
        self.layer = layer 
        self.activation = activation 
        self.bias = bias 

    def export(self): 
        return (self.id, self.layer, self.activation, self.bias) 

    @staticmethod
    def copy(g): 
        return NodeGene(g.id, g.layer, g.activation, g.bias) 

    def __str__(self): 
        return "({}, {:0.2f}, {}, {:0.3f})".format(
            self.id, 
            self.layer, 
            self.activation, 
            self.bias
        )

class ConnGene: 

    def __init__(self, id: int, enabled: bool, a: int, b: int, weight: float): 
        self.id = id 
        self.enabled = enabled 
        self.a = a 
        self.b = b 
        self.weight = weight

    def export(self): 
        return (self.id, self.enabled, self.a, self.b, self.weight)

    @staticmethod 
    def copy(g): 
        return ConnGene(g.id, g.enabled, g.a, g.b, g.weight) 

    def __str__(self): 
        return "({}, {}, {}, {}, {:0.3f})".format(
            self.id, 
            self.enabled, 
            self.a, 
            self.b, 
            self.weight 
        )

class Genome: 

    def __init__(self, nin: int, nout: int, nodes: list=[], conns: list=[]): 
        self.n_inputs = nin 
        self.n_outputs = nout 
        self.nodes = [] 
        self.conns = [] 
        self.nodes_map = {} 
        self.conns_map = {} 
        self.nodes_to_conn = {} 

        self.species = None 
        self.fitness = None 
        self.adj_fitness = None 

        for conn in conns: 
            self.add_conn(conn) 

        for node in nodes: 
            self.add_node(node) 

    def export(self): 
        return (self.n_inputs, self.n_outputs, [n.export() for n in self.nodes], [c.export() for c in self.conns])

    @staticmethod 
    def load(data): 
        return Genome(data[0], data[1], [NodeGene(*d) for d in data[2]], [ConnGene(*d) for d in data[3]])

    @staticmethod
    def copy(g): 
        return Genome(
            g.n_inputs, 
            g.n_outputs, 
            [NodeGene.copy(x) for x in g.nodes], 
            [ConnGene.copy(x) for x in g.conns] 
        )

    def __str__(self): 
        return "Genome[{}, {}, {}, {}]".format(
            self.n_inputs, 
            self.n_outputs, 
            '['+', '.join([str(x) for x in self.nodes])+']', 
            '['+', '.join([str(x) for x in self.conns])+']'
        ) 

    def add_conn(self, conn: ConnGene): 
        if conn.id in self.conns_map: 
            raise Exception("Genome already has connection with id of {}".format(conn.id))

        self.conns_map[conn.id] = conn 
        self.nodes_to_conn[(conn.a, conn.b)] = conn 
        self.conns.append(conn) 

    def add_node(self, node: NodeGene): 
        if node.id in self.nodes_map: 
            raise Exception("Genome already has node with id of {}".format(node.id))

        self.nodes_map[node.id] = node 
        self.nodes.append(node) 

    def layer(self, id: int): 
        if id < self.n_inputs: 
            return 0 
        elif id < self.n_inputs + self.n_outputs: 
            return 1 
        else: 
            return self.nodes_map[id].layer 

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
            layer = n.layer 
            if layer not in layers: 
                layers[layer] = [] 
            
            node = Node(n.id, activations[n.activation], n.activation, n.bias) 
            nodes[n.id] = node 
            layers[layer].append(node) 

        self.output_layer = [] 
        for i in range(self.n_outputs): 
            self.output_layer.append(nodes[self.n_inputs + i]) 

        for c in genome.conns: 
            enabled = c.enabled 
            if enabled: 
                a = c.a 
                b = c.b 
                w = c.weight 
                nodes[b].add_conn(nodes[a], w) 

        keys = sorted([k for k in layers]) 
        self.layers = [] 
        for k in keys: 
            self.layers.append(layers[k]) 

        self.nodes = nodes

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

    def clear(self): 
        for i in self.nodes: 
            self.nodes[i].output = 0 

    def predict(self, x): 
        
        for i in range(len(self.input_layer)):       
            node = self.input_layer[i]       
            node.output = x[i]

        for layer in self.layers: 
            for node in layer: 
                node.evaluate() 
        
        return np.array([n.output for n in self.output_layer]) 

class Species: 

    def __init__(self, start: Genome): 
        self.compare = Genome.copy(start) 
        self.genomes = [] 
        self.mean_fitness = -float('inf') 

    def update_and_sort(self): 
        if len(self.genomes) == 0: 
            self.mean_fitness = -float('inf') 
            return  

        total = 0 
        for g in self.genomes: 
            g.adj_fitness = g.fitness / len(self.genomes) 
            total += g.adj_fitness 
        self.mean_fitness = total / len(self.genomes) 
        self.sum_fitness = total 
        self.mean_true_fitness = sum(g.fitness for g in self.genomes) / len(self.genomes) 

        self.genomes.sort(key=lambda x: -x.fitness) 

def linear(x): 
    return x 

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) 

def step(x): 
    return 1.0 if x > 0.5 else 0.0 

def abs(x): 
    return np.abs(x) 

def clamp(x): 
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x 

def relu(x): 
    return max(0.0, x) 

def sin(x): 
    return np.sin(x) 

def tanh(x): 
    return np.tanh(x) 
    
class Neat: 

    args = {
        'n_pop', 
        'n_elite', 
        'clear_species', 
        'species_threshold', 
        'survive_threshold', 
        'max_species', 
        'dist_disjoint', 
        'dist_weight', 
        'dist_activation', 
        'std_mutate', 
        'std_new', 
        'prob_mutate_weight', 
        'prob_replace_weight', 
        'prob_add_conn', 
        'prob_add_node', 
        'prob_toggle_conn', 
        'prob_replace_activation', 
        'custom_activations', 
        'activations'
    }

    def __init__(self, nin: int, nout: int, args: dict={}): 
        for key in args: 
            if key not in Neat.args: 
                raise Exception("Unknown argument: {}".format(key)) 

        self.n_inputs = nin 
        self.n_outputs = nout 
        self.n_pop = args.get('n_pop', 100)  
        self.n_elite = args.get('n_elite', int(0.1 * self.n_pop)) 
        self.pop = None 
        self.species = [] 

        self.clear_species = args.get('clear_species', 5) 
        self.species_threshold = args.get('species_threshold', 1.0) 
        self.survive_threshold = args.get('survive_threshold', 0.3)  
        self.max_species = args.get('max_species', 20)  

        self.dist_disjoint = args.get('dist_disjoint', 1.0)  
        self.dist_weight = args.get('dist_weight', 0.05)  
        self.dist_activation = args.get('dist_activation', 1.0)  

        self.std_mutate_weight = args.get('std_mutate', 1.0) 
        self.std_replace_weight = args.get('std_new', 1.0) 

        self.prob_mutate_weight = args.get('prob_mutate_weight', 0.8)  
        self.prob_replace_weight = args.get('prob_replace_weight', 0.1) 
        self.prob_add_conn = args.get('prob_add_conn', 0.7)  
        self.prob_add_node = args.get('prob_add_node', 0.1)  
        self.prob_toggle_conn = args.get('prob_toggle_conn', 0.1)  
        self.prob_replace_activation = args.get('prob_replace_activation', 0.2) 

        self.gen = 0 
        self.last_fit = -float('inf') 
        self.last_inc = 0 

        activations = {
            'linear': linear, 
            'sigmoid': sigmoid, 
            'step': step,  
            'abs': abs, 
            'clamp': clamp, 
            'relu': relu, 
            'sin': sin, 
            'tanh': tanh 
        }

        extra = args.get('custom_activations') 
        if extra is not None: 
            for k in extra: 
                activations[k] = extra[k] 

        acts = args.get('activations', ['linear', 'sigmoid', 'step', 'abs', 'clamp', 'relu', 'sin', 'tanh']) 

        self.activations = {} 
        for k in acts: 
            self.activations[k] = activations[k] 

        self.conn_ids = {} 
        self.cur_conn_id = -1 
        for i in range(nin): 
            for j in range(nout): 
                self.get_conn_id(i, nin+j)   

        self.node_ids = {} 
        self.cur_node_id = nin + nout 

    def ask(self): 
        self.gen += 1 

        if self.pop is None: 
            self.pop = [self.create_genome() for _ in range(self.n_pop)] 

        return [self.create_network(g) for g in self.pop] 

    def tell(self, scores: list): 
        for s in self.species: 
            s.genomes.clear() 

        for i in range(len(self.pop)): 
            g = self.pop[i] 
            # print(g) 
            g.fitness = scores[i]
            self._add_to_species(g) 

        self.pop.sort(key=lambda x: -x.fitness) 
        for s in self.species: 
            s.update_and_sort() 

        print("Generation {}: species: {}, best - {:0.3f}, avg - {:0.3f}".format(
            self.gen, 
            len(self.species),  
            self.pop[0].fitness, 
            sum([g.fitness for g in self.pop]) / self.n_pop 
        ))

        if self.last_fit < self.pop[0].fitness: 
            self.last_fit = self.pop[0].fitness 
            self.last_inc = 0 
        else: 
            self.last_inc += 1 

        self.species.sort(key=lambda s: -s.mean_fitness)
        if len(self.species) > self.max_species: 
            self.species = [s for s in self.species[:self.max_species]] 

        if self.last_inc >= self.clear_species: 
            self.last_inc = 0 
            tmp = self.species 
            self.species = [] 
            for i in range(min(2, len(tmp))): 
                self.species.append(tmp[i]) 
            print("Taking too long to update fitness, clearing old species")

        # self.species = [s for s in self.species if self.survive_threshold * len(s.genomes) >= 0] 
        self.species = [s for s in self.species if len(s.genomes) >= 1] 

        # net = self.create_network(self.pop[0])
        # print("Best: ({:0.3f}) {}".format(net.genome.fitness, net))
        # print("- [0 0] = {:0.3f}".format(net.predict([0, 0])[0]))
        # print("- [0 1] = {:0.3f}".format(net.predict([0, 1])[0]))
        # print("- [1 0] = {:0.3f}".format(net.predict([1, 0])[0]))
        # print("- [1 1] = {:0.3f}".format(net.predict([1, 1])[0]))

        # print("Keeping best {} individuals".format(self.n_elite))

        tmp_pop = [] 
        for i in range(self.n_elite): 
            tmp_pop.append(self.pop[i]) 

        total_fit = 0.0 
        offset = 1.0 
        count = 0 

        # print("Iterating over {} species".format(len(self.species)))

        for s in self.species: 
            if len(s.genomes) == 0: 
                continue 
            if s.mean_fitness < -offset: 
                offset = -s.mean_fitness 
            total_fit += s.mean_fitness 
            count += 1 
        total_fit += offset * len(self.species) 

        n_remain = self.n_pop - len(tmp_pop) 
        total_ratio = 0.0 
        for s in self.species: 
            if len(s.genomes) == 0: 
                continue 

            ratio = None 
            if total_fit == 0: 
                ratio = 1.0 / count 
            else: 
                ratio = (s.mean_fitness + offset) / total_fit 

            # print(ratio) 

            ratio = int(ratio * n_remain) 
            
            select = max(1, int(self.survive_threshold * len(s.genomes))) 

            # print("Adding {} children from {} individuals".format(int(ratio), select))

            if select <= 0: 
                continue 

            total_ratio += ratio 

            for _ in range(int(ratio)): 
                if len(tmp_pop) == self.n_pop: 
                    print("Too many new individuals, refusing to make more") 
                    break 

                a = s.genomes[np.random.randint(0, select)] 
                b = s.genomes[np.random.randint(0, select)] 
                c = self.crossover(a, b) 
                tmp_pop.append(c) 
        
        # print("Total ratio: {}".format(total_ratio))
        # print("Adding {} random children".format(self.n_pop - len(tmp_pop)))

        while len(tmp_pop) < self.n_pop: 
            tmp_pop.append(self.create_genome())
            # if len(self.species) == 0: 
                # tmp_pop.append(self.create_genome())
            # else: 
                # s = self.species[np.random.randint(0, len(self.species))]
                # a = s.genomes[np.random.randint(0, len(s.genomes))] 
                # b = s.genomes[np.random.randint(0, len(s.genomes))] 
                # c = self.crossover(a, b) 
                # tmp_pop.append(c) 

        # for s in self.species: 
        #     print('Species: {:0.3f}, {:0.3f}, {:0.3f}, {}, n={}, c={}'.format(s.sum_fitness, s.mean_fitness, s.mean_true_fitness, len(s.genomes), len(s.compare.nodes), len(s.compare.conns)))

        self.pop = tmp_pop 

    def _add_to_species(self, g: Genome): 
        if g.species is not None: 
            g.species.genomes.append(g) 
            return 

        for s in self.species: 
            if self.distance(s.compare, g) <= self.species_threshold: 
                s.genomes.append(g) 
                return 
        s = Species(g) 
        s.genomes.append(g) 
        self.species.append(s) 

    def get_conn_id(self, a: int, b: int): 
        id = self.conn_ids.get((a, b)) 
        if id is None: 
            self.cur_conn_id += 1 
            self.conn_ids[(a, b)] = self.cur_conn_id 
            return self.cur_conn_id 
        return id 

    def get_node_id(self, cid: int): 
        id = self.node_ids.get(cid) 
        if id is None: 
            self.cur_node_id += 1 
            self.node_ids[cid] = self.cur_node_id 
            return self.cur_node_id 
        return id 

    def distance(self, a: Genome, b: Genome): 
        disjoint_conns = 0 
        weights_conns = 0
        disjoint_nodes = 0 
        weights_nodes = 0 
        act_nodes = 0 

        N_conns = min(len(a.conns), len(b.conns)) 
        N_nodes = min(len(a.nodes), len(b.nodes)) 

        ai = 0 
        bi = 0 
        while ai < len(a.conns) and bi < len(b.conns): 
            if a.conns[ai].id == b.conns[bi].id: 
                # same innovation 
                weights_conns += abs(a.conns[ai].weight - b.conns[bi].weight) 
                ai += 1 
                bi += 1 
            elif a.conns[ai].id < b.conns[bi].id: 
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
            if a.nodes[ai].id == b.nodes[bi].id: 
                # same innovation 
                weights_nodes += abs(a.nodes[ai].bias - b.nodes[bi].bias) 
                act_nodes += 0 if a.nodes[ai].activation == b.nodes[bi].activation else 1 
                ai += 1 
                bi += 1 
            elif a.nodes[ai].id < b.nodes[bi].id: 
                # add a's connection
                disjoint_nodes += 1 
                ai += 1 
            else: 
                # add b's connection 
                disjoint_nodes += 1 
                bi += 1 
        disjoint_nodes += len(a.nodes) - ai 
        disjoint_nodes += len(b.nodes) - bi 

        dist_conns = self.dist_disjoint * disjoint_conns / N_conns + self.dist_weight * weights_conns / N_conns
        dist_nodes = self.dist_disjoint * disjoint_nodes / N_nodes + self.dist_weight * weights_nodes / N_nodes + self.dist_activation * act_nodes / N_nodes

        return dist_conns + dist_nodes

    def create_genome(self): 
        nodes = [] 
        conns = [] 

        for i in range(self.n_outputs): 
            nodes.append(NodeGene(self.n_inputs+i, 1, self._new_activation(), self._new_weight())) 

        for i in range(self.n_inputs): 
            for j in range(self.n_outputs): 
                conns.append(ConnGene(self.get_conn_id(i, self.n_inputs+j), True, i, self.n_inputs+j, self._new_weight()))

        g = Genome(self.n_inputs, self.n_outputs, nodes, conns) 
        self.mutate(g) 

        return g 

    def create_network(self, g: Genome): 
        return Network(g, self.activations) 

    def crossover(self, a: Genome, b: Genome): 
        c = Genome(self.n_inputs, self.n_outputs) 

        # add all connections from A
        for conn in a.conns: 
            gene = conn 
            bconn = b.conns_map.get(gene.id) 
            if bconn is not None and self.chance(0.5):  
                gene = bconn 

            c.add_conn(ConnGene.copy(gene)) 

        # add disjoint connections from B 
        for gene in b.conns: 
            aconn = a.conns_map.get(gene.id) 
            if aconn is not None:  
                continue 

            c.add_conn(ConnGene.copy(gene)) 

        # add all nodes from A 
        for node in a.nodes: 
            gene = node 
            bnode = b.nodes_map.get(gene.id) 
            if bnode is not None and self.chance(0.5):  
                gene = bnode 

            c.add_node(NodeGene.copy(gene)) 

        # add disjoin connections from B 
        for gene in b.nodes: 
            anode = a.nodes_map.get(gene.id) 
            if anode is not None:  
                continue 

            c.add_node(NodeGene.copy(gene)) 

        self.mutate(c) 

        return c 

    def mutate(self, g: Genome): 
        # mutate connections 
        for gene in g.conns: 
            # enabled
            if self.chance(self.prob_toggle_conn): 
                gene.enabled = not gene.enabled 
            # weight 
            if self.chance(self.prob_mutate_weight): 
                gene.weight += self._mutate_weight()
            elif self.chance(self.prob_replace_weight): 
                gene.weight = self._new_weight()

        # mutate nodes 
        for gene in g.nodes: 
            # bias 
            if self.chance(self.prob_mutate_weight): 
                gene.bias += self._mutate_weight()
            elif self.chance(self.prob_replace_weight): 
                gene.bias = self._new_weight()
            # activation 
            if self.chance(self.prob_replace_activation): 
                gene.activation = self._new_activation() 

        # add node 
        if self.chance(self.prob_add_node): 
            # print("Attempting to add node") 
            for _ in range(5): 
                conn = g.conns[np.random.randint(0, len(g.conns))] 

                if not conn.enabled: 
                    continue 

                nid = self.get_node_id(conn.id) 

                # TODO might not be enough? 
                if nid in g.nodes_map: 
                    continue 

                a = conn.a 
                b = conn.b 

                a_layer = g.layer(a) 
                b_layer = g.layer(b) 
                node_layer = (a_layer + b_layer) * 0.5 

                caid = self.get_conn_id(a, nid) 
                cbid = self.get_conn_id(nid, b) 

                # print("Node {} added between {} and {}".format(nid, a, b))
                # print("Connection {} added between {} and {}".format(caid, a, nid)) 
                # print("Connection {} added between {} and {}".format(cbid, nid, b)) 

                node = NodeGene(nid, node_layer, self._new_activation(), self._new_weight()) 
                ca = ConnGene(caid, True, a, nid, self._new_weight())
                cb = ConnGene(cbid, True, nid, b, self._new_weight())
                conn.enabled = False 
                
                g.add_node(node) 
                g.add_conn(ca) 
                g.add_conn(cb) 
                break 

        # add connection 
        if self.chance(self.prob_add_conn): 
            # print("Attempting to add connection") 
            for _ in range(5): # find random nodes an arbitrary amount of times 
                a = np.random.randint(0, len(g.nodes) + self.n_inputs)
                if a >= self.n_inputs: 
                    a = g.nodes[a - self.n_inputs].id 
                b = g.nodes[np.random.randint(0, len(g.nodes))].id 

                a_layer = g.layer(a) 
                b_layer = g.layer(b) 

                # cannot add connection on same layer 
                # if a_layer == b_layer: 
                #     continue 

                # no recurrent 
                # if a_layer > b_layer: 
                #     a_layer, b_layer = b_layer, a_layer 
                #     a, b = b, a 

                # connection must not already exist 
                if (a, b) in g.nodes_to_conn: 
                    continue 

                cid = self.get_conn_id(a, b)

                # print("Connection {} added between {} and {}".format(cid, a, b)) 
                conn = ConnGene(cid, True, a, b, self._new_weight())
                g.add_conn(conn) 
                break 

    def _mutate_weight(self): 
        return np.random.randn() * self.std_mutate_weight 

    def _new_weight(self): 
        return np.random.randn() * self.std_replace_weight 

    def _new_activation(self): 
        k = [k for k in self.activations] 
        return k[np.random.randint(0, len(k))] 

    def chance(self, pct): 
        return np.random.random() < pct 

if __name__ == "__main__":
    pop = None 
    fit = None

    attempts = 1
    success = 0  
    gens = 0 

    for i in range(attempts): 
        neat = Neat(2, 1, {
            'n_pop': 500, 
            'activations': ['sigmoid'], 
            'prob_add_node': 0.3 
        }) 

        for _ in range(100): 
            pop = neat.ask() 
            fit = [] 

            for nn in pop: 
                f = 4 
                p = 2.0
                f -= np.power(np.abs(nn.predict([0, 0]) - 0), p) 
                f -= np.power(np.abs(nn.predict([0, 1]) - 1), p) 
                f -= np.power(np.abs(nn.predict([1, 0]) - 1), p) 
                f -= np.power(np.abs(nn.predict([1, 1]) - 0), p) 
                f = np.sum(f) 
                fit.append(f) 

            print('Iteration {}: '.format(i+1), end='') 
            neat.tell(fit) 

            if np.max(fit) >= 3.8: 
                print("Early stopping") 
                gens += neat.gen 
                success += 1
                break 

    print("Success rate: {:0.2f}%, Average generations: {:0.2f}".format(100 * success / attempts, gens / success))

    i = np.argmax(fit) 
    score = fit[i] 
    net = pop[i] 
    print("Score: {:0.3f}, Net: {}".format(score, net))
    print("- [0 0] = {:0.3f}".format(net.predict([0, 0])[0]))
    print("- [0 1] = {:0.3f}".format(net.predict([0, 1])[0]))
    print("- [1 0] = {:0.3f}".format(net.predict([1, 0])[0]))
    print("- [1 1] = {:0.3f}".format(net.predict([1, 1])[0]))