import time
import pickle as pkl

import networkx as nx
import numpy as np

from .random_walk import biased_random_walk

def get_rank(G, idx, method, neighbours):
    '''
    Interface function.
    G: Input Graph
    idx: index of the interested node
    method: only "indegree" is supported now
    neighbours: neighbours of idx
    '''
    if (method == "indegree"):
        indegree_dict = {}
        for idx in neighbours:
            indegree_dict[idx] = G.in_degree(idx, weight='weight')
        return indegree_dict
    else:
        raise ValueError("Invalid method!")

def get_recommended(G, num, neighbours, idx, commid, comm_dict, method="indegree", distance_limit=3):
    '''
    Get recommendations for a given question.
    G: Input Graph
    num: number of recommendations
    neighbours: list of all neighbours
    idx: idex of the questioner
    commid: the comm of the interested tag
    method: Currently only support "indegree".
    distance_limit: limit distance for BFS search
    Return list of recommendations(sorted)
    The return value is a 3-tuple (nodeid, weighted indegree, distance to target comm)
    '''
    # Compute rank of all neighbours.
    # Default method is "indegree".
    rank = get_rank(G, idx, method, neighbours)

    # Recommend based on rank and shortest path

    # First check: is there any neighbours lying in target comm
    target_nodes = []# neighbours that lie in target comm
    other_nodes = []# neighbours that *do not* lie in target comm
    
    for neighbour in neighbours:
        if (comm_dict[neighbour] == commid):
            target_nodes.append(neighbour)
        else:
            other_nodes.append(neighbour)
    
    # Get sorted rank of all target_nodes
    target_rank = {key: rank.get(key) for key in target_nodes}
    sorted_target_rank = sorted(target_rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    # If having enough nodes in target_nodes
    if (len(target_nodes) >= num):
        return [item for item in sorted_target_rank[:num]]

    # Otherwise
    # Do BFS to get the distance of each node in other_nodes toward target comm.
    result = [(*item, 0) for item in sorted_target_rank]# final result that will be returned
    num_left = num - len(result)# number of nodes that need to be selected in other_nodes
    
    distance_list = [[] for _ in range(distance_limit)]# list element i stores nodes that are i+1 steps from target comm.
    
    def search_for_comm(children,commid):# a helper function for BFS
        for child in children:
            if (comm_dict[child] == commid):
                return True
        return False

    # Do BFS in other_nodes
    for node in other_nodes:
        for distance in range(1,distance_limit):
            children = nx.descendants_at_distance(G, node, distance)
            if search_for_comm(children, commid):
                distance_list[distance-1].append(node)
                break
        else:
            distance_list[-1].append(node)

    # Append nodes to result in distance order.
    for i in range(distance_limit):
        target_rank = {key: rank.get(key) for key in distance_list[i]}
        sorted_target_rank = sorted(target_rank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        result += [(*item, i+1) for item in sorted_target_rank[:num_left]]
        if (len(result) == num):
            break
        num_left = num - len(result)
    return result

if __name__ == '__main__':
    print("Loading graph...")
    start = time.time()
    G = nx.read_gpickle("../data/weight_graph_di.pickle")
    print(f"Graph loaded in {time.time() - start:.2f} s.")

    idx = 1000

    comm = np.load("../data/community.npy")
    comm_dict = dict(zip(comm[:, 0], comm[:, 1]))

    commid = comm_dict[idx]
    print("commid: ",commid)

    print("Finding neighbours...")
    start = time.time()
    neighbours = biased_random_walk(G, idx=idx)  # threshold is an important parameter to be tuned
    print(f"Neighbours found in {time.time() - start:.2f} s.")
    print(f"{len(neighbours)} neighbours found.")
    
    with open("tmp.pkl", 'wb') as f:
        pkl.dump(neighbours,f)

    with open("tmp.pkl", 'rb') as f:
        neighbours = pkl.load(f)

    


    while (True):
        commid = int(input("Please Input comm id:"))
        print("Get recommendations...")
        start = time.time()
        recommend = get_recommended(G, num=3, neighbours=neighbours, idx=idx, commid=commid, comm_dict=comm_dict, method="indegree")
        print(recommend)
        print(f"Recommendations got in {time.time() - start:.2f} s.")
