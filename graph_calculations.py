import networkx as nx
import random
import numpy as np

def cascade_simulation(G: nx.DiGraph , seed, max_size_of_diffusion):
    """
    this function runs a simulation of a spread acording to the independent cascade (IC) model
    :param G: The network
    :type G: nx.DiGraph
    :param seed: the node that is starting the cascade
    :return: the set of infected/activated nodes
    """
    active_set = [ seed ]
    open_list = [ seed ]

    updated = True
    while len(open_list) and len(active_set) < max_size_of_diffusion:
        curr_node = open_list.pop(0)
        for neighbor in G.neighbors(curr_node):
            if neighbor not in active_set:
                if random.random() <= G.edges[ curr_node , neighbor ].get('weight'):
                    active_set.append(neighbor)
                    open_list.append(neighbor)
    return active_set

def  Atag_calc(G):
    """
    :param G: a graph, induced on the active nodes
    :return: the set of possible sources. (i.e. this function deletes from the graph all the nodes that can't reach all
    the other nodes of the active set.)
    """
    Atag=[]
    for i in G.nodes:
        #a test that ensures that all the nodes of the graph are reached from i by a directed path.
        visited = []  # List to keep track of visited nodes.
        queue = []  # Initialize a queue
        visited.append(i)
        queue.append(i)
        while len(queue)>0:
            curr = queue.pop(0)
            for neighbour in G.neighbors(curr):
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
        #a logic text, that all the nodes of G are in the list of visited, if it returns true, then i is inserted to Atag:
        visit_all_active = True
        for j in G.nodes:
            visit_all_active = visit_all_active & (j in visited)
        if visit_all_active:
            Atag.append(i)

    return Atag

def simple_reverse(G: nx.DiGraph):
    """
    this function is performing a naive reverse of the graph. and is normalising the edges of each node
    :param G: a DiGraph representing the network
    :return: a reversed Markov chain (all edges are reversed and the weights are normalized in the naive way)
    """
    simple_reversed_graph = G.reverse(copy=True)
    for v1 , v2 in simple_reversed_graph.edges:
        simple_reversed_graph.edges[ v1 , v2 ][ "weight" ] = simple_reversed_graph.edges[ v1 , v2 ][ "weight" ] / \
                                                             G.in_degree(weight="weight")[ v1 ]
    return simple_reversed_graph

def loop_reverse(G: nx.DiGraph):
    loop_reversed_graph = G.reverse(copy=True)

    #find the maximum in degree (of the original graph):
    max_in_degree = 0
    for node in G:
        node_in_degree = G.in_degree(node,weight="weight")
        if node_in_degree>max_in_degree:
            max_in_degree =node_in_degree

    #add the self loops with the wieght (max_in_dergee - node_in_degree)
    #resulting with a digraph where all the in_degrees equals to max_in_degree
    for node  in loop_reversed_graph:
        node_weight = (max_in_degree - G.in_degree(node,weight="weight"))
        loop_reversed_graph.add_edge(node,node, weight =node_weight)

    # normalize all the edges by max_in_degree:
    for v1 , v2 in loop_reversed_graph.edges:
        loop_reversed_graph.edges[ v1 , v2 ][ "weight" ] = loop_reversed_graph.edges[ v1 , v2 ][ "weight" ] / max_in_degree

    return loop_reversed_graph

def StationaryDist(G: nx.DiGraph):
    """
    retruns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: an nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
    # print("len(G.nodes):",len(G.nodes))
    mat = nx.to_numpy_array(G)
    assert(checkMarkov(mat))
    evals, evecs = np.linalg.eig(mat.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    ret_dict = {}
    if True in np.isclose(evals, 1):
        evec1 = evec1[:, 0]
        stationary = evec1 / evec1.sum()
        stationary = stationary.real
        stationary = np.array(stationary)

        node_names = []
        for n in list(G.nodes()):
            node_names.append(n)
        for n in range(len(node_names)):
            # ret_dict.update({node_names[n]:stationary[n]})
            ret_dict[node_names[n]] = stationary[n]
    else:
        print("Error in computing the stationary distribution.......")
        print("True in np.isclose(evals, 1): ",True in np.isclose(evals, 1))
    return ret_dict

def checkMarkov(m):
    """
    a function to assert that the given matrix is a Markov chain.
    (the function checks if the sum of each row is 1.)
    :param m: a matrix
    :return: bool value
    """
    for i in range(0 , len(m)):

        # Find sum of current row
        sm = 0
        for j in range(0 , len(m[ i ])):
            sm = sm + m[ i ][ j ]

        if (sm - 1>0.001):
            print("sum of line is:", sm)
            return False
    return True

def IM_based_ranking(G: nx.DiGraph , number_of_simulations=100):
    """
    a heuristic function, runs K (=number_of_simulations) independent cascade simulations from every node, and
    the probability for every node is determined by the sum of number of nodes reached by those simulations.
    :param G: a Digraph representing the network
    :param number_of_simulations: number of simulations for each node
    :return: a dict where {node: the sum of active nodes reached by node in the K simulations.}
    """
    grades = {}
    for v in G.nodes:
        sum = 0.0
        for i in range(number_of_simulations):
            a = cascade_simulation(G,v,len(list(G.nodes)))
            sum +=len(a)
        sum = sum / number_of_simulations #not necessary...
        grades[v] = sum
    return grades

def random_walk(G:nx.DiGraph, num_steps):
    '''
    this function performs a random walk estimation of a stationary distribution of a markov chain
    :param G: a DiGraph representing the network
    :param num_steps: number of steps of the random walk
    :return: a dict where  {node: number of times the random walk visited node}
    '''
    random_start = random.choice(list(G.nodes()))
    nodes_on_path = [random_start ]
    curr = random_start

    for step in range(num_steps):
        #create the list of the neighbors of curr:
        neighbors_list =[friend for friend in G.neighbors(curr)]

        #create the list of weights: for every neighbor v, we use the weight of the edge (curr,v)
        weights_list = []
        for neig in neighbors_list:
            weights_list.append(G.edges[curr,neig]['weight'])

        if len(neighbors_list) > 0:
            #random.choices returns a list of k=1 selections from neigbors_list, according to the weights_list:
            curr = random.choices(neighbors_list , weights=weights_list , k=1)[0]
            nodes_on_path.append(curr)

    #create the dict that summarise the number of visits to each node:
    ret_dict = {}
    for node in G.nodes:
        ret_dict[node] = nodes_on_path.count(node)
    return ret_dict