import os
import time
import numpy as np
import networkx as nx


if __name__ == "__main__":
    print("Loading graph...")
    start = time.time()
    OriginalGraph = nx.read_gpickle("./result/qa_graph.pickle")
    WeightedGraph = nx.read_gpickle("./result/weight_graph.pickle")
    print(f"Graph loaded in {time.time() - start:.2f} s.")

    print(f"Current number of nodes: {OriginalGraph.number_of_nodes()}")
    print(f"Current number of edges: {OriginalGraph.number_of_edges()}")

    print("Loading communities...")
    start = time.time()
    comm = np.load("./result/community.npy")
    comm_dict = dict(zip(comm[:, 0], comm[:, 1]))
    comm_unique, comm_cnt = np.unique(comm[:, 1], return_counts=True)
    comm_cnt_dict = dict(zip(comm_unique, comm_cnt))
    print(f"Communities loaded in {time.time() - start:.2f} s.")

    print(f"Current number of communities: {comm_unique.shape[0]}")
    print(f"Largest size of communities: {comm_cnt.max()}")
    print(f"Average size of communities: {comm_cnt.mean()}")

    print(f"Number of communities equal or below size 1: {np.count_nonzero(comm_cnt <= 1)}, with in total {np.sum(comm_cnt[comm_cnt <= 1])} nodes.")
    print(f"Number of communities equal or below size 3: {np.count_nonzero(comm_cnt <= 3)}, with in total {np.sum(comm_cnt[comm_cnt <= 3])} nodes.")
    print(f"Number of communities equal or below size 5: {np.count_nonzero(comm_cnt <= 5)}, with in total {np.sum(comm_cnt[comm_cnt <= 5])} nodes.")
    print(f"Number of communities equal or below size 10: {np.count_nonzero(comm_cnt <= 10)}, with in total {np.sum(comm_cnt[comm_cnt <= 10])} nodes.")
    print(f"Number of communities equal or below size 100: {np.count_nonzero(comm_cnt <= 100)}, with in total {np.sum(comm_cnt[comm_cnt <= 100])} nodes.")
    print(f"Number of communities equal or below size 1000: {np.count_nonzero(comm_cnt <= 1000)}, with in total {np.sum(comm_cnt[comm_cnt <= 1000])} nodes.")

    print(f"Number of communities above size 10000: {np.count_nonzero(comm_cnt > 10000)}, with in total {np.sum(comm_cnt[comm_cnt > 10000])} nodes.")


    print("Removing nodes in communities with size lower than 100...")
    start = time.time()
    node_list = np.random.choice(list(OriginalGraph.nodes), 10, replace=False)
    for node in node_list:
        print(f"Community for node {node} is {comm_dict[node]}")
    removed_node_list = [node for node in OriginalGraph.nodes if comm_cnt_dict[comm_dict[node]] <= 100]
    OriginalGraph.remove_nodes_from(removed_node_list)
    print(f"Nodes removed in {time.time() - start:.2f} s.")

    print(f"Current number of nodes after removal: {OriginalGraph.number_of_nodes()}")
    print(f"Current number of edges after removal: {OriginalGraph.number_of_edges()}")
