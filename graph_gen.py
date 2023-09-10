import networkx as nx
import random
def get_random_graph(nodes , edge_prob , max_diff_prob):
    """
    a function that creates a random graph (where every one of the possible edges is iinserted to the graph with probability p
    In addition, the edge probabilities (for the IC model) are selected randomly from the range [0,max_inf_prop].
    :param nodes: number of nodes
    :param edge_prob: the probability p for the selection of the possible edges
    :param max_diff_prob: the maximum value for the diffussion probabilities (that are selected from the range [0,max_diff_prob]

    :return: the resulted directed random graph, as an nx.graph
    """
    G = nx.fast_gnp_random_graph(nodes, edge_prob, directed=True)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.random() * max_diff_prob
    return G

def read_advogato_network():
    g = nx.DiGraph()
    with open(r"C:\Users\admin\PycharmProjects\SourceDetection2.0\real_graphs\out.advogato",'r') as my_file:
        for line in my_file.readlines():
            if line[0]!="%":
                line1 = line.strip().split(" ")
                v1 = int(line1[0])
                v2 = int(line1[1])
                weight1 = float(line1[2])
                g.add_edge(v_of_edge=v1,u_of_edge=v2,weight = weight1)
    print("Finished readeing Advogato network","number of nodes:",g.number_of_nodes(),"number of edges:",g.number_of_edges())
    return g

def read_network_from_file(file_path, graph_name, seperator):
    g = nx.DiGraph()
    with open(file_path ,'r') as my_file:
        for line in my_file.readlines():
            # print("line", line)
            if line[0]!="%" and line[0]!="#":
                line1 = line.split(seperator)
                if(graph_name=='epinion_trust'):
                    pass
                    #print("sep",seperator,"line1", line1)
                v1 = int(line1[0].strip())
                v2 = int(line1[1].strip())
                weight1 = random.random()
                g.add_edge(v_of_edge=v1,u_of_edge=v2,weight = weight1)
    print("Finished readeing", graph_name ,"network","number of nodes:",g.number_of_nodes(),"number of edges:",g.number_of_edges())
    return g