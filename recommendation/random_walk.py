from numpy.random import choice
import numpy as np

def biased_random_walk(G, idx):
    '''
    Calculate neighbours based on biased random walk.
    G: Input Graph
    idx: index of the interested node
    Return: list of idx's neighbourhood
    '''
    start_node = idx
    max_iter = 60
    remake_prob = 0.85
    p = 2 # 越小，越BFS
    q = 2 # 越小，越DFS
    start_node_neighbor = sorted(G.neighbors(idx))
    walk = [idx]
    curr_node = walk[0]
    curr_neighbor = sorted(G.neighbors(curr_node))
    if(len(curr_neighbor)==0):
        return None
    curr_node = choice(curr_neighbor)
    walk.append(curr_node)
    for step in range(max_iter):
        prev_node = walk[-2]
        curr_node = walk[-1]
        curr_neighbor = sorted(G.neighbors(curr_node))
        if(len(curr_neighbor)==0):
            walk.append(start_node) 
            continue
        curr_neighbor_weight = []
        for x in curr_neighbor:
            # weight = G[curr_node][x]['weight']
            if x == prev_node:
                curr_neighbor_weight.append(1/p) 
            elif G.has_edge(x,prev_node) or G.has_edge(prev_node,x):
                curr_neighbor_weight.append(1)
            else:
                curr_neighbor_weight.append(1/q)
        weight_sum = sum(curr_neighbor_weight)
        cur_neighbor_prob = [weight/weight_sum for weight in curr_neighbor_weight]
        next_node = choice(curr_neighbor,size=1,p=cur_neighbor_prob)[0]
        walk.append(next_node)
        prob = np.random.rand()
        if prob<remake_prob or step % 5 == 4:
            walk.append(start_node)
            walk.append(choice(start_node_neighbor))
            
    return list(set(walk)-set([start_node]))

if __name__=="__main__":
    # usage example
    import networkx as nx
    WeightedGraph = nx.read_gpickle("../data/weight_graph_di.pickle")
    neighbor = biased_random_walk(WeightedGraph,idx=3)
    print(sorted(neighbor),len(neighbor))
