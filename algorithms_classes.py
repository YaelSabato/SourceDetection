import networkx as nx
#my mudoles
import graph_calculations
import matplotlib as plt
import time
#these classes are holding for every algorithm the dict of probabilities for every node.
#(those probabilities are regarding the event that each node is the source of the diffusion)

class Algo():
    def __init__(self):
        self._node_dict = {}
        self._distance_from_true_source = -1
        self._number_of_success = 0
        self._total_time = 0.0
        self._Atag_size_when_success = []

    def most_probable_source_node(self, G_copy:nx.DiGraph, true_seed):
        argmax = max(self._node_dict , key=self._node_dict.get)
        self._distance_from_true_source = nx.shortest_path_length(G_copy , source=argmax , target=true_seed )
        if argmax==true_seed:
            self._number_of_success += 1
            self._Atag_size_when_success.append(len(G_copy.nodes))
        return argmax

    def get_Atag_sizes_when_success(self):
        return self._Atag_size_when_success

    def get_distance_from_true_source(self):
        return self._distance_from_true_source

    def get_number_of_success(self):
        return self._number_of_success

    def dict_reset(self):
        self._node_dict = {}


class random(Algo):
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Random                                        "
    def dict_calculation(self,  G_orig:nx.DiGraph):
        """
        this method calculates the probabilities for each node to be the true source. and assigning the dict to self._node_dict
        the only difference between the algorithms is how this dict is calculated.
        here is the default dict, which is the Random-algorithm dict. (I.e. the distribution here is the uniform distribution.
        where the probabiliti for each node in A' equals to $1/|A'|$.
        """
        t = time.time()
        n = len(G_orig.nodes)
        dict = {}
        for node in G_orig.nodes:
            dict[node] = 1.0/n
        self._node_dict = dict
        t = time.time()-t
        self._total_time += t

class Sdsd_naive(Algo): #naive (computes the stationary distribution of the naive reversed graph, without fixing the solution)
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Sdsd (naive normalization)                    "
    def dict_calculation(self, G_orig:nx.DiGraph):
        t = time.time()
        no_loops_reversed_graph = graph_calculations.simple_reverse(G_orig)
        simple_rankings_dict = graph_calculations.StationaryDist(no_loops_reversed_graph)
        self._node_dict = simple_rankings_dict
        t = time.time()-t
        self._total_time += t

class Sdsd_self_loop(Algo):  # self_loop
    def __init__(self):
        Algo.__init__(self)

    def get_name(self):
        return "Sdsd (self loops)                             "
    def dict_calculation(self, G_orig:nx.DiGraph):
        t = time.time()
        self_loops_reversed_graph = graph_calculations.loop_reverse(G_orig)
        self_loops_dict = graph_calculations.StationaryDist(self_loops_reversed_graph)
        self._node_dict = self_loops_dict
        t = time.time()-t
        self._total_time += t

class Sdsd_self_loop_with_random_walk_estimation(Algo):  # self_loop
    def __init__(self):
        Algo.__init__(self)
        self._number_of_steps = 0

    def set_number_of_steps(self,k):
        self._number_of_steps = k

    def get_name(self):
        return "Sdsd (self loops) with Random_walk_estimation K="+str(self._number_of_steps)

    def dict_calculation(self, G_orig:nx.DiGraph):
        t = time.time()
        self_loops_reversed_graph = graph_calculations.loop_reverse(G_orig)
        self_loops_dict = graph_calculations.random_walk(self_loops_reversed_graph,self._number_of_steps)
        self._node_dict = self_loops_dict
        t = time.time()-t
        self._total_time += t

class Sds_no_loop(Algo): #no_loop
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Sdsd (no loops)                               "
    def dict_calculation(self, G_orig:nx.DiGraph):
        t = time.time()
        no_loops_reversed_graph = graph_calculations.simple_reverse(G_orig)
        simple_rankings_dict = graph_calculations.StationaryDist(no_loops_reversed_graph)
        fixed_rankings_dict = {}
        for v1 in simple_rankings_dict.keys():
            # e1 = simple_rankings_dict[ v1 ]
            # e2 = G_orig.in_degree(weight="weight")[ v1 ]
            fixed_rankings_dict.update({v1: simple_rankings_dict[ v1 ] / G_orig.in_degree(weight="weight")[ v1 ]})
        self._node_dict = fixed_rankings_dict
        t = time.time()-t
        self._total_time += t

class Sds_no_loop_with_random_walk_estimation(Algo): #no_loop
    def __init__(self):
        Algo.__init__(self)
        self._number_of_steps = 0

    def set_number_of_steps(self , k):
        self._number_of_steps = k

    def get_name(self):
        return "Sdsd (no loops) with Random_walk_estimation K="+str(self._number_of_steps)
    def dict_calculation(self, G_orig:nx.DiGraph):
        t = time.time()
        no_loops_reversed_graph = graph_calculations.simple_reverse(G_orig)
        simple_rankings_dict = graph_calculations.random_walk(no_loops_reversed_graph, self._number_of_steps)
        fixed_rankings_dict = {}
        for v1 in simple_rankings_dict.keys():
            # e1 = simple_rankings_dict[ v1 ]
            # e2 = G_orig.in_degree(weight="weight")[ v1 ]
            fixed_rankings_dict.update({v1: simple_rankings_dict[ v1 ] / G_orig.in_degree(weight="weight")[ v1 ]})
        self._node_dict = fixed_rankings_dict
        t = time.time()-t
        self._total_time += t

class Max_out_deg(Algo): #max_out_degree
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Maximum out-degree                            "
    def dict_calculation(self , G_orig:nx.DiGraph):
        t = time.time()
        out_degree_dict = {}
        for n in G_orig.nodes:
            out_degree_dict.update({n: G_orig.out_degree(n)})
        self._node_dict = out_degree_dict
        t = time.time()-t
        self._total_time += t

class Min_in_deg(Algo): #min_in_degree
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Minimum in-degree                             "
    def dict_calculation(self ,  G_orig:nx.DiGraph):
        t = time.time()
        in_degree_dict = {}
        for n in G_orig.nodes:
            in_degree_dict.update({n: -1 * G_orig.in_degree(n)})
        self._node_dict = in_degree_dict
        t = time.time()-t
        self._total_time += t

class Max_out_over_in_deg(Algo):  # max{ (out_deg)/(in_deg)}
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Maximun out-degree/in-degree                  "
    def dict_calculation(self ,  G_orig:nx.DiGraph):
        t = time.time()
        out_over_in_degree_dict = {}
        for n in G_orig.nodes:
           out_over_in_degree_dict.update({n: G_orig.out_degree(n)/G_orig.in_degree(n)})
        self._node_dict = out_over_in_degree_dict
        t = time.time()-t
        self._total_time += t

class IM_based(Algo):  # IM (influence maximization) based algo
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "IM based algorithm                            "
    def dict_calculation(self , G_orig:nx.DiGraph):
        t = time.time()
        IM_grades = graph_calculations.IM_based_ranking(G_orig, 100)
        self._node_dict = IM_grades
        t = time.time()-t
        self._total_time += t

class Max_weight_arborescence(Algo):  #maximum weight arborescence (from the Italy paper: "contrasting the spread of
    # misinformatiom in online social networks" by Amoruso at. al. 2020)
    def __init__(self):
        Algo.__init__(self)
    def get_name(self):
        return "Maximum weight arborescence                   "
    def dict_calculation(self ,  G_orig:nx.DiGraph):
        t = time.time()
        max_arbo = nx.maximum_spanning_arborescence(G_orig, attr='weight')
        # print("max_arbo.edges",max_arbo.edges)
        # nx.draw_planar(max_arbo , with_labels=True)
        # plt.pyplot.savefig("max_arborescence.png")
        # plt.pyplot.show()
        # root = -1
        max_weight_arbo_dict ={}
        #The root of the arborescence is the unique node that has no incoming edges. (i.e. has indegree of 0)
        for node in max_arbo:
            # print(node,"in degree:",max_arbo.in_degree(node))
            if max_arbo.in_degree(node) == 0:
                # root = node
                max_weight_arbo_dict[node] = 1
            else:
                max_weight_arbo_dict[node] = 0
        # print(max_weight_arbo_dict)
        self._node_dict=max_weight_arbo_dict
        t = time.time()-t
        self._total_time += t