import networkx as nx

def get_cluster(G, idx):
    '''
    Interface function.
    G: Input Graph
    idx: index of the interested node
    '''
    return G.nodes[idx]["cluster"]

def get_rank(G, idx, method, neighbours):
    if (method == "indegree"):
        indegree_dict = {}
        for idx in neighbours:
            indegree_dict[idx] = G.in_degree(idx, weight='weight')
        return indegree_dict
    else:
        raise ValueError("Invalid method!")