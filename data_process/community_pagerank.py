import os
import networkx as nx
import numpy as np
import time
from tqdm import tqdm

if __name__=="__main__":
    comm_pr_path = "../data/comm_pr"

    print("Loading graph...")
    WeightedGraph = nx.read_gpickle("../data/weight_graph.pickle")
    print(f"Graph loaded.")

    print("Loading communities...")
    comm = np.load("../data/community.npy")
    comm_dict = dict(zip(comm[:, 0], comm[:, 1])) # map node id to comm id
    comm_unique, comm_cnt = np.unique(comm[:, 1], return_counts=True) # from 0 to 93491
    comm_cnt_dict = dict(zip(comm_unique, comm_cnt)) # map comm id to comm size
    print("Communities loaded.")

    big_comm_list = [comm_id for comm_id in comm_unique if comm_cnt_dict[comm_id] > 100]
    
    if not os.path.exists(comm_pr_path):
        os.mkdir(comm_pr_path)

    for comm_id in tqdm(big_comm_list):
        nodes_in_comm = comm[comm[:, 1] == comm_id][:, 0]
        nodes_out_comm = comm[comm[:, 1] != comm_id][:, 0]

        comm_graph = WeightedGraph.subgraph(nodes_in_comm)
        pagerank = nx.pagerank(comm_graph, weight="weight")

        pagerank = sorted(pagerank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        pagerank = dict(pagerank)

        np.save(os.path.join(comm_pr_path,str(comm_id)),pagerank)