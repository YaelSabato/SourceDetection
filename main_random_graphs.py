import networkx as nx
import numpy as np
import time
#my moduls:
import graph_calculations
import graph_gen
import random

import algorithms_classes

def Append_to_file(file_name, text):
    print(file_name, ":", text)
    file = open(file_name,"a")
    file.write(text + "\n")
    file.close()

def main():
    """
    this program is running a simulation of an independant cascade.
    and then runs several algorithms to find the source of the cascade,
    and compare their performance.
    """
    begin_time = time.time()
    #Our random graphs:
    #Each tuple in graph_details represent parameters of a random graph: (Graph_name,number_pf_nodes,Density,p_range)

    graph_details=[("G1",     500,   0.1,       0.0416),
                    ("G2",    1000,   0.1,       0.0204),
                    ("G3",    2000,   0.1,       0.0101),
                    ("G4",    3000,   0.1,       0.0071),
                    ("G5",    4000,   0.1,       0.0052),
                    ("G6",    5000,   0.1,       0.0041),
                    ("G7",    500,    0.0416,    0.1),
                    ("G8",    1000,   0.02,      0.1),
                    ("G9",    2000,   0.0101,    0.1),
                    ("G10",   3000,   0.0067,    0.1),
                    ("G11",   4000,   0.0052,    0.1),
                    ("G12",   5000,   0.0041,    0.1),
                    ("G13",   10000,  0.002,     0.1),
                    ("G14",   15000,  0.0013,    0.1)
                    ]

    for tuple1 in graph_details:
        # random graph parameters:
        (nodes , graph_density , edge_probability_range) = (tuple1[1] , tuple1[2] , tuple1[3])

        #output file:
        output_file = tuple1[0] +".txt"

        #difussion parameter:
        number_of_diffusions = 1000
        max_size_of_diffusion = nodes
        min_size_of_diffusion = 20

        counter_good_diff = 0
        counter_too_small_diffussions = 0
        counter_Atag_size_is_1 = 0
        sum_size_Atag = 0.0
        sum_size_Atag_other_option = 0.0

        # initiate the algorithms classes (objects the hold the dict of probabilities for each algorithm):
        sdsd_self_loop_alg = algorithms_classes.Sdsd_self_loop()
        sdsd_self_loop_RW_10 = algorithms_classes.Sdsd_self_loop_with_random_walk_estimation()
        sdsd_self_loop_RW_10.set_number_of_steps(10)
        sdsd_self_loop_RW_100 = algorithms_classes.Sdsd_self_loop_with_random_walk_estimation()
        sdsd_self_loop_RW_100.set_number_of_steps(100)
        sdsd_self_loop_RW_1000 = algorithms_classes.Sdsd_self_loop_with_random_walk_estimation()
        sdsd_self_loop_RW_1000.set_number_of_steps(1000)
        sdsd_self_loop_RW_10000 = algorithms_classes.Sdsd_self_loop_with_random_walk_estimation()
        sdsd_self_loop_RW_10000.set_number_of_steps(10000)

        sdsd_no_loop_alg = algorithms_classes.Sds_no_loop()
        sdsd_no_loop_RW_10 = algorithms_classes.Sds_no_loop_with_random_walk_estimation()
        sdsd_no_loop_RW_10.set_number_of_steps(10)
        sdsd_no_loop_RW_100 = algorithms_classes.Sds_no_loop_with_random_walk_estimation()
        sdsd_no_loop_RW_100.set_number_of_steps(100)
        sdsd_no_loop_RW_1000 = algorithms_classes.Sds_no_loop_with_random_walk_estimation()
        sdsd_no_loop_RW_1000.set_number_of_steps(1000)
        sdsd_no_loop_RW_10000 = algorithms_classes.Sds_no_loop_with_random_walk_estimation()
        sdsd_no_loop_RW_10000.set_number_of_steps(10000)
        sdsd_naive_alg = algorithms_classes.Sdsd_naive()

        random_alg = algorithms_classes.random()  # (The default algo class is the random algorithm)
        max_out_deg_alg = algorithms_classes.Max_out_deg()
        min_in_deg_alg = algorithms_classes.Min_in_deg()
        max_out_over_in_deg_alg = algorithms_classes.Max_out_over_in_deg()
        im_based_alg = algorithms_classes.IM_based()
        max_weight_arborescence_alg = algorithms_classes.Max_weight_arborescence()
        alg_list = [ sdsd_self_loop_alg ,sdsd_self_loop_RW_10, sdsd_self_loop_RW_100, sdsd_self_loop_RW_1000,sdsd_self_loop_RW_10000,\
                     sdsd_no_loop_alg ,sdsd_no_loop_RW_10,sdsd_no_loop_RW_100,sdsd_no_loop_RW_1000,sdsd_no_loop_RW_10000,\
                     sdsd_naive_alg ,max_weight_arborescence_alg , random_alg , max_out_deg_alg ,\
                     min_in_deg_alg , max_out_over_in_deg_alg , im_based_alg ]

        # create or read the network:
        #Random graph:
        # g = graph_gen.get_random_graph(nodes , edge_prob=graph_density , max_diff_prob=edge_probability_range)
        #advogato graph:
        g = graph_gen.read_advogato_network()
        # print("g:",g)
        # print("g.edges[:10]:",list(g.edges)[:10])

        while counter_good_diff<number_of_diffusions:
            #select a random seed:
            rand_seed = random.choice(list(g.nodes()))
            # print("rand_seed:",rand_seed)
            #run a diffusion simulation, recieving a set of active nodes:
            active_list = graph_calculations.cascade_simulation(g , rand_seed,max_size_of_diffusion)
            # print("len(active_list):",len(active_list))
            # print("active_list:",active_list)
            if len(active_list)< min_size_of_diffusion:
                print("too small diffusion. len(active set)= ", len(active_list))
                counter_too_small_diffussions += 1
            if len(active_list) >= min_size_of_diffusion:
                counter_good_diff +=1
                #firstly, shuffle the nodes to prevent a bais based on the order of the nodes in active_list
                random.shuffle(active_list)
                # print("active_list (after shuffle):",active_list)
                #clean from the graph all the not-active nodes:
                G_copy = g.subgraph(active_list)
                # find the irreducible subgraph. (that is, all the nodes v such that there is a path in the graph from v to
                # every other active node.)
                Atag = graph_calculations.Atag_calc(G_copy)
                print("len(Atag): ",len(Atag))
                if len(Atag)==1:
                    print("|A'| is 1, so there is only one possible source")
                    counter_Atag_size_is_1 += 1
                else:
                    if len(Atag) < len(active_list):
                        G_copy = g.subgraph(Atag)
                    sum_size_Atag +=len(Atag)
                    sum_size_Atag_other_option +=(1/len(Atag))
                    print("\n*****************************\n")
                    print("iteration number:",counter_good_diff)
                    for alg1 in alg_list :
                        print("\nname of current algo: ",alg1.get_name())
                        # erase the dict from previous iteretions:
                        alg1.dict_reset()
                        # calculate dict for the current iteretion:
                        alg1.dict_calculation(G_orig=G_copy)
                        print(alg1._node_dict)
                        print("most probable source node:",alg1.most_probable_source_node(G_copy,rand_seed),"true seed: ",rand_seed)
                        print(alg1.get_name() , "    : " , alg1.get_number_of_success())
                    print("_________summary of iteretion %d_________" % counter_good_diff)
                    for alg in alg_list:
                        print(alg.get_name() , alg.get_number_of_success(), alg.get_Atag_sizes_when_success())
                    print("______________________________________")
        total_time = time.time()-begin_time
        print("_____________________END________________________")

        Append_to_file(file_name=output_file, text="resultd for the graph %s (nodes=%d,density=%f,p_range=%f)" % tuple1 )
        Append_to_file(file_name=output_file, text="total time (minuts):"+ str(total_time/60))
        for alg in alg_list:
            Append_to_file(file_name=output_file, text=alg.get_name()+":"+str(alg.get_number_of_success())+"\ttotal time:"+str(alg._total_time)+"\tAtag sizes"+str(alg.get_Atag_sizes_when_success()))

        Append_to_file(file_name=output_file, text="number of to small diffusions: "+str(counter_too_small_diffussions))
        Append_to_file(file_name=output_file, text="number of times where |A'| was 1 (so all the algorithms are basically the same):"+str(counter_Atag_size_is_1 ))
        Append_to_file(file_name=output_file, text="number of 'good' diffusions:"+str(counter_good_diff))
        average = (sum_size_Atag/counter_good_diff)
        Append_to_file(file_name=output_file, text="average size of A':"+str(average))
        Append_to_file(file_name=output_file, text="average size of A' (computed by adding 1/|A'| in every iteration):"+str( sum_size_Atag_other_option))
        Append_to_file(file_name=output_file, text="____________________________\n")
if __name__ == '__main__':
    main()











    #*************************************************************
    # average_A = sum(list_len_active_set)/len(list_len_active_set)
    # print("list_len_active_set", list_len_active_set)
    # median_A = statistics.median(list_len_active_set)
    # average_A_tag = sum(list_len_Atag) /len (list_len_Atag)
    # print("list_len_Atag",list_len_Atag)
    # median_A_tag = statistics.median(list_len_Atag)

    # number_selfloop = fixed_reg_output[0]
    # number_simple = simple_reg_output[0]
    # number_random = random_reg_output[0]
    # number_average = average_reg_output[0]
    # number_perm = perm_reg_output[0]
    # number_maxoutdeg = max_out_reg_output[0]
    # number_minindeg = min_in_reg_output[0]
    # number_out_over_in_deg = max_OutOverIn_reg_output[0]
    # number_IM = IM_reg_output[0]
    # # number_basicMCMC = basicMCMC_reg_output[0]

    # average_edges = sum(list_number_of_edges)/len(list_number_of_edges)
    # print("list_number_of_edges (orig graph)",list_number_of_edges)
    # average_edges_active_set = sum(list_number_of_edges_in_active_set)/len(list_number_of_edges_in_active_set)
    # print("list_number_of_edges_in_active_set",list_number_of_edges_in_active_set)
    # average_edges_Atag = sum(list_number_of_edges_in_Atag)/len(list_number_of_edges_in_Atag)
    # print("list_number_of_edges_in_Atag",list_number_of_edges_in_Atag)

    # a = ""+str(number_selfloop)
    # b = ""+str(number_simple)
    # e = "" + number_random
    # c= ""+number_of_times_len_Atag_is_1
    #
    # text1 = diff_details +","+str(average_edges)+","+str(average_A)+","+str(average_edges_active_set)+","+str(average_A_tag)+","+str(average_edges_Atag)+","+str(median_A)+","+str(median_A_tag)
    # text2 = text1 +","+str(number_selfloop)+","+str(number_simple)+","+str(number_average)+","+str(number_perm)+","\
    #         +str(number_random)+","+str(number_maxoutdeg)+","+str(number_minindeg)+","+str(number_out_over_in_deg)+","\
    #         +str(number_IM)+","+str(number_of_times_len_Atag_is_1)+","+str(number_of_to_small_diff)

#","+number_basicMCMC+

    # Append_to_file(file_name, text2)
    # plot_the_output(fig_file_name, simple_hist,fixed_hist,random_hist,text2)






