import networkx as nx
import numpy as np

def _load_comm(filename):
    '''
    a closure to maintain loded data in utils.py
    '''
    comm = np.load(filename)
    comm_dict = dict(zip(comm[:, 0], comm[:, 1]))
    def inner(G, idx):
        '''
        Interface function.
        G: Input Graph
        idx: index of the interested node
        '''
        return comm_dict[idx]
    return inner

get_cluster = _load_comm("../data/community.npy")

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